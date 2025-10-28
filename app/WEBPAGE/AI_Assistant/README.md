# Sage AI Assistant for Sentinel Diagnostics

## Overview
Sage is an intelligent AI assistant powered by Google's Gemini Flash 2.0 model, designed specifically for the Sentinel Diagnostics healthcare platform. It helps users navigate the platform, understand features, and provides guidance for both patients and doctors.

## Features
- **Role-aware assistance**: Different responses for patients vs doctors
- **Healthcare-focused**: Trained on platform-specific knowledge
- **Real-time chat**: WebSocket-powered instant messaging
- **Quick suggestions**: Context-appropriate quick action buttons
- **Knowledge search**: Search platform documentation and guides
- **Conversation history**: Maintains context across chat sessions

## Setup

### 1. Environment Variables
Create or update the `.env` file in the AI_Assistant directory:
```bash
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

### 2. Install Dependencies
```bash
cd WEBPAGE/AI_Assistant
pip install -r requirements.txt
```

### 3. Integration with Main App
Add these lines to your main `app.py`:

```python
# Import AI Assistant blueprint
from AI_Assistant.routes import ai_assistant_bp

# Register the blueprint
app.register_blueprint(ai_assistant_bp)
```

### 4. Knowledge Base
The assistant uses `knowledge.txt` (located in project root) as its knowledge base. This file contains comprehensive platform documentation and user guides.

## API Endpoints

### POST `/api/sage/chat`
Main chat endpoint for conversations with Sage.

**Request:**
```json
{
    "message": "How do I book an appointment?",
    "user_role": "patient",
    "conversation_history": [
        {"role": "user", "content": "Previous message"},
        {"role": "assistant", "content": "Previous response"}
    ]
}
```

**Response:**
```json
{
    "success": true,
    "response": "To book an appointment, go to the Appointments page...",
    "user_role": "patient",
    "timestamp": "2025-10-17T10:30:00.000Z"
}
```

### GET `/api/sage/suggestions`
Get quick suggestion buttons.

**Query Params:**
- `role`: "patient" or "doctor"

**Response:**
```json
{
    "success": true,
    "suggestions": [
        "How do I book an appointment?",
        "How to take the lung cancer risk assessment?",
        "How do I submit a medical case?",
        "How can I view my reports?",
        "How to join a video consultation?"
    ],
    "user_role": "patient"
}
```

### POST `/api/sage/search`
Search the knowledge base.

**Request:**
```json
{
    "query": "appointment booking"
}
```

**Response:**
```json
{
    "success": true,
    "results": "Relevant knowledge base content...",
    "query": "appointment booking"
}
```

### GET `/api/sage/health`
Health check endpoint.

**Response:**
```json
{
    "success": true,
    "status": "healthy",
    "message": "Sage AI Assistant is running"
}
```

## Frontend Integration

The chat interface is already integrated into the patient layout (`layouts/patient.html`). The chat widget includes:

- **Toggle Button**: Fixed position button to open/close chat
- **Chat Container**: Full chat interface with messages
- **Input Area**: Message input with send button
- **Auto-scroll**: Messages automatically scroll to bottom
- **Keyboard Support**: Enter key to send messages

### JavaScript Integration Example:
```javascript
// Send message to Sage
async function sendMessageToSage(message) {
    try {
        const response = await fetch('/api/sage/chat', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({
                message: message,
                user_role: 'patient', // or 'doctor'
                conversation_history: getConversationHistory()
            })
        });
        
        const data = await response.json();
        
        if (data.success) {
            displayMessage('sage', data.response);
        } else {
            displayMessage('error', data.error || 'Something went wrong');
        }
    } catch (error) {
        console.error('Error sending message:', error);
    }
}
```

## Customization

### Modifying Sage's Personality
Edit the `_create_system_prompt()` method in `functions.py` to adjust Sage's personality, tone, and behavior.

### Adding New Features
1. Add new methods to the `SageAssistant` class
2. Create corresponding routes in `routes.py`
3. Update the frontend JavaScript to use new endpoints

### Updating Knowledge Base
Simply edit the `knowledge.txt` file in the project root. The assistant will automatically use the updated content.

## Error Handling

The assistant includes comprehensive error handling:
- **API Key Missing**: Graceful degradation with helpful error messages
- **Model Unavailable**: Fallback responses when Gemini is down
- **Invalid Requests**: Proper validation and error responses
- **Network Issues**: Timeout handling and retry logic

## Security Considerations

1. **API Key Protection**: Never commit API keys to version control
2. **Input Validation**: All user inputs are sanitized
3. **Rate Limiting**: Consider implementing rate limiting for production
4. **Session Management**: Integrate with existing Flask session handling

## Development

### Testing
Run the test suite:
```bash
python test_assistant.py
```

### Debugging
Enable debug logging by setting the logging level:
```python
logging.basicConfig(level=logging.DEBUG)
```

## Production Deployment

1. **Environment Variables**: Set proper production API keys
2. **Rate Limiting**: Implement request rate limiting
3. **Monitoring**: Add health check monitoring
4. **Caching**: Consider caching responses for common queries
5. **Load Balancing**: Scale horizontally if needed

## Troubleshooting

### Common Issues

**"GEMINI_API_KEY not found"**
- Ensure `.env` file exists with valid API key
- Check environment variable is properly loaded

**"Model not responding"** 
- Verify internet connection
- Check Gemini API status
- Validate API key permissions

**"Knowledge base not found"**
- Ensure `knowledge.txt` exists in project root
- Check file permissions and encoding (UTF-8)

### Logs
Check application logs for detailed error information:
```bash
tail -f app.log
```

## Support
For issues specific to Sage AI Assistant, check:
1. Application logs
2. API endpoint health checks
3. Gemini API status page
4. Network connectivity

## Version History
- **v1.0**: Initial implementation with Gemini Flash 2.0
- Integration with Sentinel Diagnostics platform
- Role-aware patient/doctor assistance
- Real-time chat interface