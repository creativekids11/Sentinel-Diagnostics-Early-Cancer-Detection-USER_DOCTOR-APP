# Sage AI Assistant - Integration Guide

## 🚀 Quick Setup

### 1. Prerequisites
```bash
# Install required packages
cd WEBPAGE/AI_Assistant
pip install -r requirements.txt
```

### 2. Environment Configuration
Create or update `.env` file in `WEBPAGE/AI_Assistant/`:
```env
GEMINI_API_KEY=your_actual_gemini_api_key_here
```

> **Get your API key**: Visit [Google AI Studio](https://makersuite.google.com/app/apikey) to get your Gemini API key

### 3. Integration Steps

#### Step 1: Blueprint Registration (Already Done)
The AI Assistant is automatically registered in `app.py`:
```python
# AI Assistant is already integrated
from AI_Assistant.routes import ai_assistant_bp
app.register_blueprint(ai_assistant_bp)
```

#### Step 2: Frontend Integration (Already Done)
The chat widget is included in `layouts/patient.html`:
```html
<!-- Sage AI Assistant is already integrated -->
{% include 'AI_Assistant/chat_widget.html' %}
```

#### Step 3: Test the Integration
```bash
# Test the assistant directly
cd WEBPAGE/AI_Assistant
python demo.py

# Run the full app
cd WEBPAGE
python app.py
```

---

## 🎯 Usage Guide

### For Patients:
1. **Access**: Click the "Sage AI Assistant" button on any patient page
2. **Chat**: Type questions about using the platform
3. **Quick Help**: Use suggestion buttons for common questions
4. **Examples**:
   - "How do I book an appointment?"
   - "What is the lung cancer risk assessment?"
   - "How do I view my medical reports?"

### For Doctors (Future):
1. **Access**: Same chat interface with doctor-specific context
2. **Examples**:
   - "How do I approve patient cases?"
   - "How to use the AI scanner?"
   - "How do I generate medical reports?"

---

## 🔧 API Endpoints

### Chat API
```http
POST /api/sage/chat
Content-Type: application/json

{
    "message": "How do I book an appointment?",
    "user_role": "patient",
    "conversation_history": [...]
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

### Suggestions API
```http
GET /api/sage/suggestions?role=patient
```

**Response:**
```json
{
    "success": true,
    "suggestions": [
        "How do I book an appointment?",
        "How to take the lung cancer risk assessment?",
        "How do I submit a medical case?"
    ]
}
```

### Health Check
```http
GET /api/sage/health
```

---

## 🎨 Customization

### 1. Modify Sage's Personality
Edit `AI_Assistant/functions.py`:
```python
def _create_system_prompt(self):
    return f"""You are Sage, the helpful AI assistant...
    
    Your personality:
    - Friendly and empathetic
    - Healthcare-focused
    - [Add your customizations here]
    """
```

### 2. Update Knowledge Base
Edit `knowledge.txt` in the project root. Changes are automatically loaded.

### 3. Customize Chat Interface
Edit `AI_Assistant/chat_widget.html` to modify:
- Colors and styling
- Chat bubble design  
- Animation effects
- Button layouts

### 4. Add New Features
1. Add methods to `SageAssistant` class
2. Create new routes in `routes.py`
3. Update frontend JavaScript

---

## 🔍 Troubleshooting

### Common Issues

#### "GEMINI_API_KEY not found"
```bash
# Solution: Set your API key
echo "GEMINI_API_KEY=your_key_here" > WEBPAGE/AI_Assistant/.env
```

#### "Chat widget not appearing"
- Check browser console for JavaScript errors
- Ensure you're on a patient page (not doctor page yet)
- Verify chat widget is included in template

#### "AI responses are slow"
- Normal for first request (model loading)
- Subsequent responses should be faster
- Check your internet connection

#### "Import errors"
```bash
# Solution: Install dependencies
pip install google-generativeai python-dotenv flask
```

### Debug Mode
Enable detailed logging:
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

---

## 🚀 Advanced Configuration

### 1. Model Parameters
Adjust in `functions.py`:
```python
self.model = genai.GenerativeModel(
    model_name="gemini-2.0-flash-exp",
    generation_config={
        "temperature": 0.7,  # Creativity (0-1)
        "top_p": 0.8,        # Diversity (0-1)
        "max_output_tokens": 1024,  # Response length
    }
)
```

### 2. Rate Limiting
Add to `routes.py`:
```python
from flask_limiter import Limiter

limiter = Limiter(
    app,
    key_func=lambda: session.get('user_id'),
    default_limits=["100 per hour"]
)

@ai_assistant_bp.route('/chat', methods=['POST'])
@limiter.limit("10 per minute")
def sage_chat():
    # ... existing code
```

### 3. Response Caching
```python
from functools import lru_cache

@lru_cache(maxsize=100)
def get_cached_response(message_hash):
    # Cache common responses
    pass
```

---

## 📊 Monitoring & Analytics

### 1. Usage Tracking
Add to database:
```sql
CREATE TABLE sage_interactions (
    id INTEGER PRIMARY KEY,
    user_id INTEGER,
    message TEXT,
    response TEXT,
    timestamp DATETIME,
    user_role TEXT
);
```

### 2. Performance Metrics
```python
import time

start_time = time.time()
response = assistant.get_response(message)
response_time = time.time() - start_time

# Log metrics
logger.info(f"Response time: {response_time:.2f}s")
```

---

## 🔐 Security Considerations

### 1. Input Validation
- All user inputs are sanitized
- XSS protection in chat interface
- SQL injection prevention

### 2. Rate Limiting
- Prevent API abuse
- User-based limits
- IP-based fallback

### 3. Data Privacy
- No sensitive data in logs
- Conversation history limits
- Secure API key storage

---

## 🚀 Production Deployment

### 1. Environment Variables
```bash
export GEMINI_API_KEY="your_production_key"
export FLASK_SECRET="your_secure_secret_key"
export FLASK_ENV="production"
```

### 2. Performance Optimization
- Enable response caching
- Use CDN for static assets
- Implement load balancing

### 3. Monitoring
- Health check endpoints
- Error tracking (Sentry)
- Performance monitoring (New Relic)

---

## 📈 Future Enhancements

### Planned Features:
- [ ] Doctor-specific chat interface
- [ ] Voice input/output
- [ ] Multi-language support
- [ ] Integration with medical databases
- [ ] Appointment booking via chat
- [ ] Report generation assistance

### Extension Ideas:
- Medical symptom checker
- Drug interaction warnings
- Health tips and reminders
- Integration with wearable devices

---

## 🆘 Support

### Getting Help:
1. **Documentation**: Check this guide and README.md
2. **Issues**: Test with `demo.py` first
3. **API Problems**: Check `/api/sage/health` endpoint
4. **Frontend Issues**: Check browser console

### Contact:
- Check application logs for detailed errors
- Verify API key permissions
- Test with minimal examples first

---

**Last Updated**: October 17, 2025  
**Version**: 1.0.0  
**Compatibility**: Sentinel Diagnostics Platform