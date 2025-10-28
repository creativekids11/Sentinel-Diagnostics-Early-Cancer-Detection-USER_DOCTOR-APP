# AI Medical Platform - Complete Deployment Guide

## 🎯 Overview

This guide provides comprehensive instructions for deploying the AI Medical Diagnosis Platform using our streamlined 2-script deployment process.

## 📋 Prerequisites

### System Requirements
- **OS:** Windows 10/11 (64-bit)
- **RAM:** Minimum 8GB (16GB recommended for AI models)
- **Storage:** 10GB free space
- **Network:** WiFi/Ethernet connection for network sharing

### Software Requirements

#### 1. Python 3.8+
```bash
# Check if Python is installed
python --version

# If not installed, download from: https://python.org/downloads/
# ⚠️ IMPORTANT: Check "Add Python to PATH" during installation
```

#### 2. Nginx Web Server
```bash
# Download nginx for Windows: http://nginx.org/en/download.html
# Extract to: C:\nginx\
# Verify installation:
C:\nginx\nginx.exe -v
```

## 🚀 Deployment Process

### Phase 1: Initial Deployment

1. **Open PowerShell as Administrator**
   ```bash
   Right-click PowerShell → "Run as administrator"
   cd "d:\deployed-hackathon"
   ```

2. **Run Main Deployment Script**
   ```bash
   .\GO.bat
   ```

   **What GO.bat does:**
   - ✅ Stops any running services
   - ✅ Installs Python dependencies (Flask, PyTorch, OpenCV, etc.)
   - ✅ Configures nginx reverse proxy
   - ✅ Starts AI medical platform
   - ✅ Loads AI models (Brain, Lung, Breast cancer detection)
   - ✅ Tests all services
   - ✅ Shows access URLs

### Phase 2: Network Configuration

3. **Configure Network Sharing (Run as Administrator)**
   ```bash
   Right-click admin-network-fix.bat → "Run as administrator"
   ```

   **What admin-network-fix.bat does:**
   - ✅ Adds Windows Firewall rules for ports 80 and 5000
   - ✅ Enables network discovery and file sharing
   - ✅ Sets network profile to Private
   - ✅ Configures ICMP (ping) for diagnostics
   - ✅ Shows network IP addresses for sharing

## 🌐 Access Points

After successful deployment:

### Local Access (Your Computer)
- **URL:** http://localhost
- **Purpose:** Personal testing and administration

### Network Access (Colleagues)
- **URL:** http://[your-ip-address]
- **Purpose:** Share with colleagues on same WiFi
- **Note:** IP address shown after running admin-network-fix.bat

## 🏥 Platform Features

### AI Diagnosis Capabilities
1. **Brain Tumor Detection**
   - Upload brain MRI images
   - Get segmentation results
   - View tumor boundaries and analysis

2. **Lung Cancer Detection**
   - YOLO-based detection system
   - Real-time analysis
   - Confidence scoring

3. **Breast Cancer Analysis**
   - MLMO-EMO algorithm
   - Multi-objective optimization
   - Comprehensive reporting

4. **AI Medical Assistant**
   - Natural language consultation
   - Medical knowledge base
   - Patient guidance

### Web Interface Features
- **Patient Portal:** Registration, login, medical history
- **Doctor Dashboard:** Review, approve, manage cases
- **Report Generation:** Automated medical reports
- **Payment System:** Consultation fee processing
- **Chat Interface:** AI assistant integration

## 🔧 Troubleshooting

### Common Issues & Solutions

#### Issue: GO.bat fails with "Python not found"
**Solution:**
```bash
# Add Python to PATH manually
setx PATH "%PATH%;C:\Python39;C:\Python39\Scripts"
# Restart PowerShell and try again
```

#### Issue: "Nginx failed to start"
**Solutions:**
1. Check nginx installation:
   ```bash
   C:\nginx\nginx.exe -t -c "d:\deployed-hackathon\nginx-windows.conf"
   ```

2. Check if port 80 is in use:
   ```bash
   netstat -ano | findstr :80
   # If occupied, stop the process or change port in nginx-windows.conf
   ```

3. Run as Administrator:
   ```bash
   Right-click GO.bat → "Run as administrator"
   ```

#### Issue: Network access not working
**Solutions:**
1. Ensure admin-network-fix.bat was run as Administrator
2. Check Windows Firewall:
   ```bash
   # Temporarily disable to test
   netsh advfirewall set allprofiles state off
   # Re-enable after testing
   netsh advfirewall set allprofiles state on
   ```

3. Verify same network:
   ```bash
   # From colleague's computer, ping your IP
   ping [your-ip-address]
   ```

#### Issue: AI models not loading
**Solutions:**
1. Check available memory (AI models need 4-8GB RAM)
2. Install missing dependencies:
   ```bash
   python -m pip install torch torchvision opencv-python pillow numpy pandas
   ```

3. Check logs:
   ```bash
   type app.log | findstr ERROR
   ```

### Advanced Troubleshooting

#### Port Configuration
If port 80 is occupied, modify `nginx-windows.conf`:
```nginx
server {
    listen 8080;  # Change from 80 to 8080
    # ... rest of config
}
```

#### Virtual Environment (Optional)
For isolated Python environment:
```bash
python -m venv venv
venv\Scripts\activate
python -m pip install -r requirements-compatible.txt
python production_wsgi_enhanced.py
```

#### Manual Service Start
If GO.bat fails, start services manually:
```bash
# Start nginx
C:\nginx\nginx.exe -c "d:\deployed-hackathon\nginx-windows.conf"

# Start Python app
python production_wsgi_enhanced.py
```

## 📊 Performance Optimization

### For Better Performance:
1. **Close unnecessary applications** before running AI models
2. **Use SSD storage** for faster model loading
3. **Increase virtual memory** if RAM is limited
4. **Use wired connection** for stable network sharing

### Resource Usage:
- **CPU:** 2-4 cores recommended
- **RAM:** 8GB minimum, 16GB optimal
- **GPU:** Optional but improves AI model performance
- **Network:** 100Mbps for smooth multi-user access

## 🛡️ Security Considerations

### Data Privacy:
- All processing is **local** - no data leaves your network
- Medical images stored temporarily in secure folders
- Session data encrypted and auto-expires
- No external API calls for AI processing

### Network Security:
- Platform accessible only on local network
- Firewall rules restrict to specific ports
- Admin-level network configuration required
- Automatic session timeouts implemented

### Recommended Security Practices:
1. **Use strong passwords** for doctor accounts
2. **Regularly update** Python packages
3. **Monitor access logs** in app.log
4. **Backup patient data** regularly
5. **Restrict admin access** to authorized personnel

## 🔄 Maintenance

### Regular Tasks:
```bash
# Check service status
tasklist | findstr "nginx.exe python.exe"

# View recent logs
powershell "Get-Content app.log -Tail 20"

# Update Python packages
python -m pip install --upgrade pip
python -m pip install --upgrade -r requirements-compatible.txt

# Restart services
.\GO.bat
```

### Updates & Patches:
1. **Backup current installation** before updates
2. **Test updates** in development environment first
3. **Update AI models** when new versions available
4. **Monitor performance** after updates

## 📞 Support & Contact

### Self-Help Resources:
1. **Log Files:** Check `app.log` for detailed error messages
2. **Configuration:** Review `nginx-windows.conf` for web server settings
3. **Dependencies:** Check `requirements-compatible.txt` for package versions

### Getting Help:
1. **Document the error:** Copy exact error messages
2. **Check prerequisites:** Verify Python and nginx installation
3. **Try as Administrator:** Run both scripts with admin privileges
4. **Contact team:** Provide system info and error logs

### Development Team:
- **Devansh Madake** - Lead Developer
- **Jay Hatapaki** - Backend Systems
- **Ujjwal Arora** - Frontend & UX
- **Pranad Nair** - AI Models
- **Kanderp Thakore** - DevOps

---

## 🎉 Success Checklist

After deployment, verify these work:
- [ ] http://localhost loads the main page
- [ ] Brain tumor detection accepts image uploads
- [ ] Lung cancer detection processes correctly
- [ ] Breast cancer analysis functions
- [ ] AI assistant responds to questions
- [ ] Doctor login works
- [ ] Patient registration functions
- [ ] Network access works from colleague's device
- [ ] All static assets (CSS, JS, images) load properly
- [ ] No errors in app.log

**🎯 When all items are checked, your AI Medical Platform is ready for production use!**