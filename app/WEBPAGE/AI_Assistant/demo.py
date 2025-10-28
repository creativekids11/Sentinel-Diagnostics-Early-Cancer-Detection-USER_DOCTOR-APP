#!/usr/bin/env python3
"""
Demo script for Sage AI Assistant
Demonstrates the functionality without running the full Flask app
"""

import os
import sys
from dotenv import load_dotenv

# Add current directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

def main():
    print("🤖 SAGE AI ASSISTANT - DEMO")
    print("=" * 40)
    
    # Load environment variables
    load_dotenv()
    
    # Check for API key
    api_key = os.getenv('GEMINI_API_KEY')
    if not api_key or api_key in ['your_gemini_api_key_here', 'test_api_key_123']:
        print("❌ No valid GEMINI_API_KEY found!")
        print("   Please set your Gemini API key in the .env file")
        print("   Example: GEMINI_API_KEY=your_actual_api_key_here")
        return
    
    print(f"🔑 API Key found: {api_key[:10]}...")
    
    try:
        from functions import SageAssistant, create_assistant_response
        
        print("\n🧪 Testing Sage Assistant creation...")
        assistant = SageAssistant()
        print("✅ Sage Assistant created successfully")
        
        print(f"📚 Knowledge base loaded: {len(assistant.knowledge_base)} characters")
        
        # Interactive chat loop
        print("\n💬 INTERACTIVE CHAT MODE")
        print("=" * 40)
        print("Type 'quit' to exit, 'suggestions' to see quick suggestions")
        print("You can specify role with 'patient:' or 'doctor:' prefix")
        print("Example: 'patient: How do I book an appointment?'\n")
        
        conversation_history = []
        current_role = "patient"
        
        while True:
            try:
                user_input = input(f"[{current_role}] You: ").strip()
                
                if user_input.lower() in ['quit', 'exit', 'bye']:
                    print("👋 Goodbye! Thanks for testing Sage!")
                    break
                
                if user_input.lower() == 'suggestions':
                    suggestions = assistant.get_quick_suggestions(current_role)
                    print(f"\n💡 Quick suggestions for {current_role}s:")
                    for i, suggestion in enumerate(suggestions, 1):
                        print(f"  {i}. {suggestion}")
                    print()
                    continue
                
                # Check for role prefix
                if ':' in user_input:
                    parts = user_input.split(':', 1)
                    role_prefix = parts[0].strip().lower()
                    if role_prefix in ['patient', 'doctor']:
                        current_role = role_prefix
                        user_input = parts[1].strip()
                        print(f"🔄 Switched to {current_role} mode")
                
                if not user_input:
                    continue
                
                print("🤖 Sage is thinking...")
                
                # Get response
                response = assistant.get_response(
                    user_message=user_input,
                    user_role=current_role,
                    conversation_history=conversation_history[-10:]  # Last 10 messages
                )
                
                if response['success']:
                    print(f"🤖 Sage: {response['response']}\n")
                    
                    # Update conversation history
                    conversation_history.append({
                        'role': 'user',
                        'content': user_input
                    })
                    conversation_history.append({
                        'role': 'assistant', 
                        'content': response['response']
                    })
                else:
                    print(f"❌ Error: {response.get('error', 'Unknown error')}\n")
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted by user. Goodbye!")
                break
            except Exception as e:
                print(f"❌ Error: {e}")
                continue
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("   Make sure you have installed the required packages:")
        print("   pip install -r requirements.txt")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_api_functions():
    """Test the API functions"""
    print("\n🧪 TESTING API FUNCTIONS")
    print("=" * 40)
    
    try:
        from functions import create_assistant_response
        
        test_messages = [
            ("How do I register for Sentinel Diagnostics?", "patient"),
            ("How do I approve patient cases?", "doctor"),
            ("What is the lung cancer risk assessment?", "patient"),
            ("How do I use the AI scanner?", "doctor"),
        ]
        
        for message, role in test_messages:
            print(f"\n📨 Testing: [{role}] {message}")
            response = create_assistant_response(message, role)
            
            if response['success']:
                print(f"✅ Success: {response['response'][:100]}...")
            else:
                print(f"❌ Failed: {response.get('error', 'Unknown error')}")
        
    except Exception as e:
        print(f"❌ API test failed: {e}")

def show_knowledge_stats():
    """Show knowledge base statistics"""
    print("\n📊 KNOWLEDGE BASE STATISTICS")
    print("=" * 40)
    
    try:
        from functions import SageAssistant
        
        assistant = SageAssistant()
        knowledge = assistant.knowledge_base
        
        lines = knowledge.split('\n')
        words = knowledge.split()
        
        print(f"📄 Total lines: {len(lines)}")
        print(f"🔤 Total words: {len(words)}")
        print(f"📝 Total characters: {len(knowledge)}")
        
        # Count sections
        sections = [line for line in lines if line.startswith('##')]
        print(f"📑 Sections: {len(sections)}")
        
        # Key topics
        key_topics = ['patient', 'doctor', 'appointment', 'case', 'report', 'ai', 'diagnosis']
        print(f"\n🔍 Key topic mentions:")
        for topic in key_topics:
            count = knowledge.lower().count(topic)
            print(f"  • {topic.title()}: {count} times")
        
    except Exception as e:
        print(f"❌ Knowledge stats failed: {e}")

if __name__ == "__main__":
    main()
    
    # Additional tests
    if len(sys.argv) > 1 and sys.argv[1] == '--test':
        test_api_functions()
        show_knowledge_stats()