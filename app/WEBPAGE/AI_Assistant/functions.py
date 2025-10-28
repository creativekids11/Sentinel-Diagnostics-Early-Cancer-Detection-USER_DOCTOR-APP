import os
import google.generativeai as genai
from dotenv import load_dotenv
import json
import logging
from typing import Any, Dict, List, Optional

# Load environment variables
load_dotenv()

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SageAssistant:
    LANGUAGE_MAPPING = {
        'hi': 'hi-IN',  # Hindi
        'mr': 'mr-IN',  # Marathi
        'kn': 'kn-IN',  # Kannada
        'ml': 'ml-IN',  # Malayalam
        'pa': 'pa-IN',  # Punjabi
        'en': 'en-US'   # English
    }

    LANGUAGE_LABELS = {
        'en-US': 'English',
        'hi-IN': 'Hindi',
        'mr-IN': 'Marathi',
        'kn-IN': 'Kannada',
        'ml-IN': 'Malayalam',
        'pa-IN': 'Punjabi'
    }

    DEFAULT_LANGUAGE = 'en-US'

    def __init__(self):
        """Initialize the Sage AI Assistant with Gemini Flash 2.0"""
        self.api_key = os.getenv('GEMINI_API_KEY')
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
        
        # Configure Gemini
        genai.configure(api_key=self.api_key)
        
        # Initialize the model (Gemini Flash 2.0)
        self.model = genai.GenerativeModel(
            model_name="gemini-2.0-flash-exp",
            generation_config={
                "temperature": 0.7,
                "top_p": 0.8,
                "top_k": 40,
                "max_output_tokens": 1024,
            }
        )
        
        # Load knowledge base
        self.knowledge_base = self._load_knowledge_base()
        
        # Initialize chat session with system prompt
        self.system_prompt = self._create_system_prompt()
        
    def _load_knowledge_base(self) -> str:
        """Load the knowledge base from knowledge.txt"""
        try:
            # Try multiple possible paths
            possible_paths = [
                "knowledge.txt",
                "WEBPAGE/AI_Assistant/knowledge.txt", 
                "../knowledge.txt",
                "AI_Assistant/knowledge.txt"
            ]
            
            for path in possible_paths:
                if os.path.exists(path):
                    with open(path, 'r', encoding='utf-8') as file:
                        return file.read()
            
            # If no file found, return default knowledge
            logger.warning("Knowledge base file not found, using default")
            return self._get_default_knowledge()
            
        except Exception as e:
            logger.error(f"Error loading knowledge base: {e}")
            return self._get_default_knowledge()
    
    def _get_default_knowledge(self) -> str:
        """Return default knowledge if file not found"""
        return """
        You are Sage, an AI assistant for Sentinel Diagnostics healthcare platform.
        Help users navigate the app and answer healthcare-related questions.
        
        Key features:
        - Patient registration and login
        - Doctor appointments booking
        - AI-powered cancer risk assessments  
        - Medical case submissions
        - Video consultations
        - Medical report generation
        
        Always be helpful, empathetic, and direct users to appropriate healthcare professionals for medical advice.
        """
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for Sage"""
        return f"""You are Sage, the helpful AI assistant for Sentinel Diagnostics healthcare platform. 

Your personality:
- Friendly, empathetic, and professional
- Healthcare-focused but not providing medical diagnoses
- Knowledgeable about the platform's features
- Always encourage users to consult healthcare professionals for medical advice

Your knowledge base:
{self.knowledge_base}

Guidelines:
1. Help users navigate the platform (registration, appointments, features)
2. Answer questions about platform functionality
3. Provide step-by-step instructions when needed
4. Be empathetic when discussing health concerns
5. Always remind users that you're an assistant, not a doctor
6. Direct users to appropriate platform features
7. Keep responses concise but helpful
8. Use emojis sparingly but appropriately

Remember: You assist with platform usage, not medical diagnosis. Always encourage consulting healthcare professionals for medical concerns."""

    def get_response(self, user_message: str, user_role: str = "patient", conversation_history: List[Dict] = None, detected_language: str = None) -> Dict:
        """
        Get response from Sage AI Assistant with persistent language detection
        
        Args:
            user_message: The user's message (in any language)
            user_role: "patient" or "doctor"  
            conversation_history: Previous conversation messages
            detected_language: Previously detected language (if any)
            
        Returns:
            Dict with response and metadata
        """
        try:
            language_info = None
            # Step 1: Use stored language or detect new one
            if detected_language and detected_language != 'auto-detect':
                current_lang = detected_language
                language_label = self._get_language_label(current_lang)
                language_info = {
                    "language_code": current_lang,
                    "language_label": language_label,
                    "base_language": current_lang.split('-')[0],
                    "confidence": None,
                    "source": "stored"
                }
                logger.info(f"Using stored language: {current_lang}")
            else:
                language_info = self.detect_language(user_message)
                language_info["source"] = "detected"
                current_lang = language_info["language_code"]
                language_label = language_info["language_label"]
                logger.info(
                    "Detected new language: %s (base=%s, confidence=%s) for message: %s...",
                    current_lang,
                    language_info.get("base_language"),
                    language_info.get("confidence"),
                    user_message[:50]
                )
            
            # Fallback if detection did not set label
            if not language_info:
                language_label = self._get_language_label(self.DEFAULT_LANGUAGE)
                language_info = {
                    "language_code": self.DEFAULT_LANGUAGE,
                    "language_label": language_label,
                    "base_language": 'en',
                    "confidence": None,
                    "source": "default"
                }
            else:
                language_label = language_info.get("language_label", self._get_language_label(current_lang))
            
            # Step 2: Translate user message to English for AI processing
            english_message = self._translate_to_english(user_message, current_lang)
            logger.info(f"Translated user input to English: {english_message[:100]}...")
            
            # Step 3: Build conversation context with English
            conversation_context = self._build_conversation_context(
                english_message, user_role, conversation_history
            )
            
            # Step 4: Generate response in English
            chat = self.model.start_chat(history=[])
            english_response = chat.send_message(conversation_context)
            english_response_text = english_response.text.strip()
            logger.info(f"AI English response: {english_response_text[:100]}...")
            
            # Step 5: Translate response to the detected/stored language
            final_response = self._translate_from_english(english_response_text, current_lang)
            logger.info(f"Translated to {current_lang}: {final_response[:100]}...")
            
            # Verify translation worked (not just English fallback)
            if final_response == english_response_text and not current_lang.startswith('en'):
                logger.warning(f"Translation to {current_lang} failed, returned English text")
                final_response = f"[Translation failed] {english_response_text}"
            
            # Format response
            return {
                "success": True,
                "response": final_response,
                "detected_language": current_lang,
                "original_language": current_lang,
                "language_label": language_label,
                "language_confidence": language_info.get("confidence"),
                "language_base": language_info.get("base_language"),
                "language_source": language_info.get("source"),
                "english_message": english_message,
                "english_response": english_response_text,
                "user_role": user_role,
                "timestamp": self._get_timestamp()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                "success": False,
                "response": "I'm sorry, I'm having trouble processing your request right now. Please try again in a moment.",
                "error": str(e),
                "detected_language": self.DEFAULT_LANGUAGE,
                "original_language": self.DEFAULT_LANGUAGE,
                "language_label": self._get_language_label(self.DEFAULT_LANGUAGE),
                "language_confidence": None,
                "language_base": 'en',
                "language_source": "error",
                "timestamp": self._get_timestamp()
            }
    
    def _build_conversation_context(self, user_message: str, user_role: str, conversation_history: List[Dict]) -> str:
        """Build the full conversation context for the AI"""
        
        context = f"{self.system_prompt}\n\n"
        context += f"Current user role: {user_role}\n\n"
        
        # Add conversation history if provided
        if conversation_history:
            context += "Previous conversation:\n"
            for msg in conversation_history[-5:]:  # Last 5 messages for context
                role = "User" if msg.get("role") == "user" else "Sage"
                context += f"{role}: {msg.get('content', '')}\n"
            context += "\n"
        
        context += f"Current user message: {user_message}\n\n"
        context += "Respond as Sage in English:"
        
        return context
    
    def _get_timestamp(self) -> str:
        """Get current timestamp"""
        from datetime import datetime
        return datetime.now().isoformat()
    
    def _get_language_label(self, language_code: str) -> str:
        return self.LANGUAGE_LABELS.get(language_code, self.LANGUAGE_LABELS[self.DEFAULT_LANGUAGE])

    def detect_language(self, text: str) -> Dict[str, Any]:
        """Detect the language of the input text using Gemini"""
        try:
            # Use Gemini for language detection
            prompt = f"What language is this text written in? Answer with only the language name (choose from: English, Hindi, Marathi, Kannada, Malayalam, Punjabi). If unsure, say 'English':\n\n{text}"
            response = self.model.generate_content(prompt)
            detected_name = response.text.strip().lower()
            logger.info(f"Gemini detected language: '{detected_name}' for text: {text[:50]}...")
            
            # Map detected name to our codes
            name_to_code = {
                'english': 'en-US',
                'hindi': 'hi-IN', 
                'marathi': 'mr-IN',
                'kannada': 'kn-IN',
                'malayalam': 'ml-IN',
                'punjabi': 'pa-IN'
            }
            
            language_code = name_to_code.get(detected_name, self.DEFAULT_LANGUAGE)
            language_label = self._get_language_label(language_code)
            
            return {
                "language_code": language_code,
                "language_label": language_label,
                "base_language": language_code.split('-')[0],
                "confidence": None,  # Gemini doesn't provide confidence
                "source": "gemini"
            }
        except Exception as e:
            logger.warning(f"Gemini language detection failed: {e}, defaulting to English")
            return {
                "language_code": self.DEFAULT_LANGUAGE,
                "language_label": self._get_language_label(self.DEFAULT_LANGUAGE),
                "base_language": 'en',
                "confidence": None,
                "source": "error"
            }

    def _detect_language(self, text: str) -> str:
        """Backward-compatible helper returning only the language code"""
        return self.detect_language(text)["language_code"]
    
    def _translate_to_english(self, text: str, source_lang: str) -> str:
        """Translate text to English using Gemini if needed"""
        try:
            # If already English, return as is
            if source_lang.startswith('en'):
                return text
            
            # Get language name for better Gemini understanding
            lang_name = self._get_language_label(source_lang)
            
            # Use Gemini for translation
            prompt = f"Translate the following text from {lang_name} to English. Only return the translated text, no explanations:\n\n{text}"
            response = self.model.generate_content(prompt)
            translated = response.text.strip()
            return translated if translated else text
            
        except Exception as e:
            logger.warning(f"Translation to English failed: {e}, using original text")
            return text
    
    def _translate_from_english(self, text: str, target_lang: str) -> str:
        """Translate text from English to target language using Gemini"""
        try:
            # If target is English, return as is
            if target_lang.startswith('en'):
                return text
            
            # Get language name for better Gemini understanding
            lang_name = self._get_language_label(target_lang)
            
            # Use Gemini for translation
            prompt = f"Translate the following English text to {lang_name}. Only return the translated text, no explanations:\n\n{text}"
            response = self.model.generate_content(prompt)
            translated = response.text.strip()
            return translated if translated else text
            
        except Exception as e:
            logger.warning(f"Translation from English failed: {e}, using English text")
            return text
    
    def get_quick_suggestions(self, user_role: str) -> List[str]:
        """Get quick suggestion buttons based on user role"""
        if user_role == "doctor":
            return [
                "How do I approve patient cases?",
                "How to use the AI scanner?", 
                "How do I generate medical reports?",
                "How to set my availability status?",
                "How to start a video consultation?"
            ]
        else:  # patient
            return [
                "How do I book an appointment?",
                "How to take the lung cancer risk assessment?",
                "How do I submit a medical case?",
                "How can I view my reports?",
                "How to join a video consultation?"
            ]
    
    def search_knowledge(self, query: str) -> str:
        """Search the knowledge base for specific information"""
        try:
            # Simple keyword search in knowledge base
            lines = self.knowledge_base.lower().split('\n')
            query_words = query.lower().split()
            
            relevant_lines = []
            for line in lines:
                if any(word in line for word in query_words):
                    relevant_lines.append(line.strip())
            
            if relevant_lines:
                return '\n'.join(relevant_lines[:5])  # Return top 5 matches
            else:
                return "No specific information found in knowledge base."
                
        except Exception as e:
            logger.error(f"Error searching knowledge: {e}")
            return "Unable to search knowledge base at the moment."

# Global instance
sage_assistant = None

def get_sage_assistant() -> SageAssistant:
    """Get or create the Sage assistant instance"""
    global sage_assistant
    if sage_assistant is None:
        sage_assistant = SageAssistant()
    return sage_assistant

def create_assistant_response(user_message: str, user_role: str = "patient", conversation_history: List[Dict] = None, detected_language: str = None) -> Dict:
    """
    Main function to get response from Sage Assistant
    
    Args:
        user_message: User's input message
        user_role: "patient" or "doctor"
        conversation_history: List of previous messages
        detected_language: Previously detected language (if any)
        
    Returns:
        Dict with AI response and metadata
    """
    try:
        assistant = get_sage_assistant()
        return assistant.get_response(user_message, user_role, conversation_history, detected_language)
    except Exception as e:
        logger.error(f"Error in create_assistant_response: {e}")
        return {
            "success": False,
            "response": "I'm currently unavailable. Please try again later or contact support.",
            "error": str(e)
        }