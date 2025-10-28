# AI Medical Diagnosis Platform

A comprehensive AI-powered medical diagnosis platform that detects and analyzes:
- 🧠 **Brain Tumor Detection** with segmentation
- 🫁 **Lung Cancer Detection** using YOLO models  
- 🎗️ **Breast Cancer Analysis** with MLMO-EMO algorithm
- 🤖 **AI Medical Assistant** for patient consultation
- 📊 **Medical Report Generation** and patient management

## 🚀 Quick Start (2 Minutes)

### Prerequisites
- Windows 10/11
- Python 3.8+ installed and in PATH
- Nginx installed at `C:\nginx\` ([Download here](http://nginx.org/en/download.html))

### Step 1: Deploy the Platform
```bash
.\GO.bat
```
This single command will:
- Install all Python dependencies
- Configure nginx reverse proxy
- Start the AI medical platform
- Test all services
- Show access URLs

### Step 2: Fix Network Sharing (Run as Administrator)
```bash
Right-click → Run as Administrator
.\admin-network-fix.bat
```
This will configure Windows firewall and network settings for colleague access.

### Step 3: Access the Platform
- **Local access:** http://localhost
- **Network access:** http://[your-ip-address] (shown after deployment)

## 🔧 Troubleshooting

### If GO.bat fails:
1. Ensure Python is in your PATH: `python --version`
2. Ensure nginx is installed at `C:\nginx\nginx.exe`
3. Run as Administrator if permission errors occur
4. Check logs in `app.log` for detailed error information

### If network access doesn't work:
1. Run `admin-network-fix.bat` as Administrator
2. Check Windows Firewall settings
3. Ensure you're on the same WiFi network as colleagues
4. Try disabling Windows Defender temporarily

## 📁 Project Structure

```
AI-Medical-Platform/
├── GO.bat                      # Main deployment script
├── admin-network-fix.bat       # Network configuration (run as admin)
├── production_wsgi_enhanced.py # Main application server
├── nginx-windows.conf          # Nginx configuration
├── app/                       # Main application code
│   ├── WEBPAGE/               # Web interface
│   ├── models_lungcancer_questionary/ # Lung cancer models
│   ├── MLMO-EMO_algorithm_otsu/       # Breast cancer algorithm
│   └── Lung_Cancer/           # Lung cancer detection
├── instance/                  # Flask instance folder
└── logs/                     # Application logs
```

## 🎯 Features

### Medical AI Capabilities
- **Brain Tumor Segmentation:** Advanced deep learning models for tumor detection and boundary delineation
- **Lung Cancer YOLO Detection:** Real-time object detection for lung cancer identification
- **Breast Cancer MLMO-EMO:** Multi-objective optimization algorithm for enhanced breast cancer analysis
- **AI Medical Assistant:** Intelligent chatbot for patient consultation and medical guidance

### Web Platform Features
- **Patient Management:** Registration, login, and medical history tracking
- **Doctor Dashboard:** Professional interface for medical review and approval
- **Medical Reports:** Automated report generation with AI analysis results
- **Payment System:** Integrated payment processing for medical consultations
- **Real-time Chat:** AI-powered medical assistant with contextual responses

## 🔒 Security & Privacy

- All medical data is processed locally
- No data is sent to external servers
- HIPAA-compliant data handling practices
- Secure session management
- Encrypted data transmission

## 📞 Support

If you encounter issues:
1. Check the `app.log` file for detailed error messages
2. Ensure all prerequisites are properly installed
3. Try running both scripts as Administrator
4. Contact the development team with specific error messages

## 🏆 Team

**Hackathon 2.0 Development Team:**
- **Devansh Madake** - Lead Developer, Medical AI Models, DevOps & Deployment, Data Science & AI Architecture
- **Jay Hatapaki** - Frontend and Backend Development & Integration
- **Ujjwal Arora** - Model Training and Computational Resources
- **Pranad Nair** - UI/UX Devlopment
- **Kanderp Thakore** - Designing and Nutrition Planner

---

**⚡ Ready to revolutionize medical diagnosis with AI? Run `.\GO.bat` now!**
