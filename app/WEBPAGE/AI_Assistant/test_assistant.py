#!/usr/bin/env python3
"""
Test script for Sage AI Assistant
Tests the core functionality of the Gemini Flash 2.0 integration
"""

import unittest
import os
import sys
from unittest.mock import patch, Mock

# Add the current directory to the path to import our modules
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from functions import SageAssistant, create_assistant_response, get_sage_assistant

class TestSageAssistant(unittest.TestCase):
    """Test cases for Sage AI Assistant"""

    def setUp(self):
        """Set up test environment"""
        # Mock environment variables
        os.environ['GEMINI_API_KEY'] = 'test_api_key_123'
        
    def tearDown(self):
        """Clean up after tests"""
        # Clean up environment
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_sage_assistant_initialization(self, mock_model, mock_configure):
        """Test that SageAssistant initializes correctly"""
        
        # Mock the model
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        try:
            assistant = SageAssistant()
            
            # Verify Gemini was configured
            mock_configure.assert_called_once_with(api_key='test_api_key_123')
            
            # Verify model was created
            mock_model.assert_called_once()
            
            # Check that knowledge base was loaded
            self.assertIsInstance(assistant.knowledge_base, str)
            self.assertGreater(len(assistant.knowledge_base), 0)
            
            print("✅ Sage Assistant initialization test passed")
            
        except Exception as e:
            self.fail(f"SageAssistant initialization failed: {e}")
    
    def test_missing_api_key(self):
        """Test behavior when API key is missing"""
        
        # Remove API key
        if 'GEMINI_API_KEY' in os.environ:
            del os.environ['GEMINI_API_KEY']
        
        with self.assertRaises(ValueError) as context:
            SageAssistant()
        
        self.assertIn("GEMINI_API_KEY not found", str(context.exception))
        print("✅ Missing API key test passed")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_knowledge_base_loading(self, mock_model, mock_configure):
        """Test knowledge base loading functionality"""
        
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        assistant = SageAssistant()
        
        # Check knowledge base content
        knowledge = assistant.knowledge_base
        self.assertIsInstance(knowledge, str)
        
        # Should contain key platform information
        self.assertIn("Sentinel Diagnostics", knowledge)
        self.assertIn("patient", knowledge.lower())
        self.assertIn("doctor", knowledge.lower())
        
        print("✅ Knowledge base loading test passed")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_quick_suggestions(self, mock_model, mock_configure):
        """Test quick suggestions for different user roles"""
        
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        assistant = SageAssistant()
        
        # Test patient suggestions
        patient_suggestions = assistant.get_quick_suggestions("patient")
        self.assertIsInstance(patient_suggestions, list)
        self.assertGreater(len(patient_suggestions), 0)
        
        # Should contain appointment-related suggestion
        suggestions_text = ' '.join(patient_suggestions).lower()
        self.assertIn("appointment", suggestions_text)
        
        # Test doctor suggestions
        doctor_suggestions = assistant.get_quick_suggestions("doctor")
        self.assertIsInstance(doctor_suggestions, list)
        self.assertGreater(len(doctor_suggestions), 0)
        
        # Should contain case-related suggestion
        doctor_text = ' '.join(doctor_suggestions).lower()
        self.assertIn("case", doctor_text)
        
        print("✅ Quick suggestions test passed")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_knowledge_search(self, mock_model, mock_configure):
        """Test knowledge base search functionality"""
        
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        
        assistant = SageAssistant()
        
        # Search for appointment information
        results = assistant.search_knowledge("appointment booking")
        self.assertIsInstance(results, str)
        
        # Search for non-existent information
        no_results = assistant.search_knowledge("nonexistent query xyz123")
        self.assertIsInstance(no_results, str)
        
        print("✅ Knowledge search test passed")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_get_response_structure(self, mock_model, mock_configure):
        """Test that get_response returns correct structure"""
        
        # Mock the model and chat
        mock_chat_instance = Mock()
        mock_response = Mock()
        mock_response.text = "This is a test response from Sage."
        mock_chat_instance.send_message.return_value = mock_response
        
        mock_model_instance = Mock()
        mock_model_instance.start_chat.return_value = mock_chat_instance
        mock_model.return_value = mock_model_instance
        
        assistant = SageAssistant()
        
        # Test response structure
        response = assistant.get_response("How do I book an appointment?", "patient")
        
        # Check response structure
        self.assertIsInstance(response, dict)
        self.assertIn("success", response)
        self.assertIn("response", response)
        self.assertIn("user_role", response)
        self.assertIn("timestamp", response)
        
        # Check values
        self.assertTrue(response["success"])
        self.assertEqual(response["user_role"], "patient")
        self.assertIsInstance(response["response"], str)
        
        print("✅ Response structure test passed")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_conversation_history_handling(self, mock_model, mock_configure):
        """Test conversation history is properly handled"""
        
        # Mock the model and chat
        mock_chat_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Response with history context."
        mock_chat_instance.send_message.return_value = mock_response
        
        mock_model_instance = Mock()
        mock_model_instance.start_chat.return_value = mock_chat_instance
        mock_model.return_value = mock_model_instance
        
        assistant = SageAssistant()
        
        # Test with conversation history
        history = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi! How can I help?"},
            {"role": "user", "content": "I need help with appointments"}
        ]
        
        response = assistant.get_response("What are the steps?", "patient", history)
        
        # Should still return proper structure
        self.assertIsInstance(response, dict)
        self.assertTrue(response["success"])
        
        # Verify send_message was called (meaning context was built)
        mock_chat_instance.send_message.assert_called_once()
        
        print("✅ Conversation history test passed")
    
    @patch('google.generativeai.configure')
    @patch('google.generativeai.GenerativeModel')
    def test_create_assistant_response_function(self, mock_model, mock_configure):
        """Test the main create_assistant_response function"""
        
        # Mock the model and chat
        mock_chat_instance = Mock()
        mock_response = Mock()
        mock_response.text = "Test response from create_assistant_response."
        mock_chat_instance.send_message.return_value = mock_response
        
        mock_model_instance = Mock()
        mock_model_instance.start_chat.return_value = mock_chat_instance
        mock_model.return_value = mock_model_instance
        
        # Test the main function
        response = create_assistant_response("How do I register?", "patient")
        
        self.assertIsInstance(response, dict)
        self.assertIn("success", response)
        
        print("✅ create_assistant_response function test passed")


def run_manual_tests():
    """Run manual tests that require actual API calls (optional)"""
    print("\n" + "="*50)
    print("MANUAL TESTS (require valid GEMINI_API_KEY)")
    print("="*50)
    
    # Check if real API key is available
    real_api_key = os.getenv('GEMINI_API_KEY')
    
    if not real_api_key or real_api_key == 'test_api_key_123' or real_api_key == 'your_gemini_api_key_here':
        print("❌ No valid GEMINI_API_KEY found in environment")
        print("   Set a real API key to run manual tests")
        return False
    
    try:
        print(f"🔑 Using API key: {real_api_key[:10]}...")
        
        # Test real assistant creation
        print("🧪 Testing real Sage Assistant creation...")
        assistant = SageAssistant()
        print("✅ Sage Assistant created successfully")
        
        # Test real response generation
        print("🧪 Testing real response generation...")
        response = assistant.get_response("Hello! How do I book an appointment?", "patient")
        
        if response.get("success"):
            print("✅ Real response generated successfully")
            print(f"📝 Response: {response['response'][:100]}...")
        else:
            print(f"❌ Response generation failed: {response.get('error', 'Unknown error')}")
            return False
        
        # Test suggestions
        print("🧪 Testing real suggestions...")
        suggestions = assistant.get_quick_suggestions("patient")
        print(f"✅ Generated {len(suggestions)} suggestions")
        
        return True
        
    except Exception as e:
        print(f"❌ Manual test failed: {e}")
        return False


def main():
    """Run all tests"""
    print("🤖 SAGE AI ASSISTANT - TEST SUITE")
    print("="*40)
    
    # Run unit tests
    print("\n🧪 RUNNING UNIT TESTS...")
    unittest.main(argv=[''], exit=False, verbosity=2)
    
    # Run manual tests if API key is available
    print("\n🔍 CHECKING FOR MANUAL TESTS...")
    manual_success = run_manual_tests()
    
    print("\n" + "="*40)
    print("📊 TEST SUMMARY")
    print("="*40)
    print("✅ Unit tests: See results above")
    print(f"{'✅' if manual_success else '❌'} Manual tests: {'PASSED' if manual_success else 'SKIPPED/FAILED'}")
    
    if not manual_success:
        print("\n💡 TIP: To run manual tests, set a valid GEMINI_API_KEY environment variable")
    
    print("\n🎉 Test suite completed!")


if __name__ == "__main__":
    main()