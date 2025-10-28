from flask import Blueprint, request, jsonify, session
from .functions import create_assistant_response, get_sage_assistant
import logging

# Create blueprint for AI Assistant routes
ai_assistant_bp = Blueprint('ai_assistant', __name__, url_prefix='/api/sage')

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@ai_assistant_bp.route('/chat', methods=['POST'])
def sage_chat():
    """
    Main chat endpoint for Sage AI Assistant
    
    Expected JSON payload:
    {
        "message": "User's message",
        "conversation_history": [previous messages] (optional),
        "detected_language": "previously detected language" (optional)
    }
    """
    try:
        # Get request data
        data = request.get_json()
        if not data or 'message' not in data:
            return jsonify({
                "success": False,
                "error": "Message is required"
            }), 400
        
        user_message = data['message'].strip()
        if not user_message:
            return jsonify({
                "success": False, 
                "error": "Message cannot be empty"
            }), 400
        
        # Get user role from session (default to patient)
        user_role = "patient"  # Default
        if 'user_id' in session:
            # You can get actual user role from database here if needed
            # For now, we'll assume patient unless specified
            user_role = data.get('user_role', 'patient')
        
        # Get conversation history
        conversation_history = data.get('conversation_history', [])
        
        # Get previously detected language from session or request
        detected_language = session.get('sage_detected_language') or data.get('detected_language')
        
        # Get AI response with persistent language
        response = create_assistant_response(
            user_message=user_message,
            user_role=user_role,
            conversation_history=conversation_history,
            detected_language=detected_language
        )
        
        # Store the detected language in session for future requests
        if response.get('success') and response.get('detected_language'):
            session['sage_detected_language'] = response['detected_language']
            logger.info(f"Stored detected language in session: {response['detected_language']}")
        
        return jsonify(response)
        
    except Exception as e:
        logger.error(f"Error in sage_chat: {e}")
        return jsonify({
            "success": False,
            "error": "Internal server error",
            "message": "I'm having trouble right now. Please try again in a moment."
        }), 500

@ai_assistant_bp.route('/detect-language', methods=['POST'])
def detect_language():
    """Detect the language of a provided text snippet"""
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({
                "success": False,
                "error": "Text is required"
            }), 400

        text = data['text'].strip()
        if not text:
            return jsonify({
                "success": False,
                "error": "Text cannot be empty"
            }), 400

        assistant = get_sage_assistant()
        detection = assistant.detect_language(text)

        return jsonify({
            "success": True,
            "language_code": detection.get("language_code"),
            "language_label": detection.get("language_label"),
            "base_language": detection.get("base_language"),
            "confidence": detection.get("confidence")
        })

    except Exception as e:
        logger.error(f"Error detecting language: {e}")
        return jsonify({
            "success": False,
            "error": "Language detection failed"
        }), 500

@ai_assistant_bp.route('/suggestions', methods=['GET'])
def get_suggestions():
    """
    Get quick suggestion buttons for the user
    """
    try:
        # Get user role from query params or session
        user_role = request.args.get('role', 'patient')
        
        # Get suggestions from assistant
        assistant = get_sage_assistant()
        suggestions = assistant.get_quick_suggestions(user_role)
        
        return jsonify({
            "success": True,
            "suggestions": suggestions,
            "user_role": user_role
        })
        
    except Exception as e:
        logger.error(f"Error getting suggestions: {e}")
        return jsonify({
            "success": False,
            "error": str(e),
            "suggestions": []
        }), 500

@ai_assistant_bp.route('/search', methods=['POST'])
def search_knowledge():
    """
    Search the knowledge base
    
    Expected JSON payload:
    {
        "query": "Search query"
    }
    """
    try:
        data = request.get_json()
        if not data or 'query' not in data:
            return jsonify({
                "success": False,
                "error": "Query is required"
            }), 400
        
        query = data['query'].strip()
        if not query:
            return jsonify({
                "success": False,
                "error": "Query cannot be empty"
            }), 400
        
        # Search knowledge base
        assistant = get_sage_assistant()
        results = assistant.search_knowledge(query)
        
        return jsonify({
            "success": True,
            "results": results,
            "query": query
        })
        
    except Exception as e:
        logger.error(f"Error searching knowledge: {e}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@ai_assistant_bp.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint for the AI Assistant"""
    try:
        # Test if assistant can be initialized
        assistant = get_sage_assistant()
        return jsonify({
            "success": True,
            "status": "healthy",
            "message": "Sage AI Assistant is running"
        })
    except Exception as e:
        logger.error(f"Health check failed: {e}")
        return jsonify({
            "success": False,
            "status": "unhealthy", 
            "error": str(e)
        }), 500