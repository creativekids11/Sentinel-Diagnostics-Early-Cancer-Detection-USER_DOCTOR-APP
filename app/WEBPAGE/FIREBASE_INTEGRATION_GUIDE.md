# Firebase Google OAuth Integration Guide for Sentinel Diagnostics

## Overview
This integration adds Google OAuth authentication to Sentinel Diagnostics while preserving the existing username/password authentication system and doctor verification process.

## Features Implemented

### 1. **Dual Authentication System**
- Existing username/password authentication preserved
- New Google OAuth login/signup option added
- Seamless integration with existing user database

### 2. **Google Profile Integration**
- Automatic profile photo download from Google accounts
- Profile photos saved to existing `static/uploads/photos/` directory
- Fallback to default avatar if photo download fails

### 3. **Doctor Verification for Google Signups**
- Special verification flow for doctors using Google OAuth
- Requires qualification certificates and documents
- PAN card verification (if enabled)
- Admin approval process maintained

### 4. **Enhanced User Experience**
- Clean UI with Google OAuth buttons in login/signup forms
- Role selection modal for Google signups
- Progress indicators and error handling

## Setup Instructions

### 1. **Firebase Project Setup**
1. Go to [Firebase Console](https://console.firebase.google.com/)
2. Create a new project or use existing one
3. Enable Authentication with Google provider
4. Add your domain to authorized domains
5. Copy your Firebase configuration

### 2. **Update Firebase Configuration**
Edit `static/js/firebase-config.js` and replace the configuration:

```javascript
const firebaseConfig = {
    apiKey: "your-actual-api-key",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id",
    storageBucket: "your-project.appspot.com",
    messagingSenderId: "123456789",
    appId: "your-app-id"
};
```

### 3. **Database Migration**
The `google_id` column will be automatically added to the users table when the app starts. No manual migration needed.

### 4. **Environment Configuration**
Ensure your `.env` file includes any necessary configurations:

```env
# Your existing environment variables
# Add any Firebase-related configs if needed
```

## Technical Implementation

### Files Modified/Added:

#### **Templates**
- `templates/auth/login.html` - Added Google OAuth button and JavaScript
- `templates/auth/signup.html` - Added Google OAuth with role selection modal
- `templates/auth/doctor_verification.html` - New verification page for Google doctor signups

#### **Backend Routes**
- `/auth/google/login` - Handle Google OAuth login
- `/auth/google/signup` - Handle Google OAuth signup with role selection
- `/doctor/verification/<user_id>` - Doctor verification form
- `/doctor/verification/<user_id>` (POST) - Process doctor verification

#### **Database Schema**
- Added `google_id` column to users table for linking Google accounts

#### **Frontend Assets**
- `static/js/firebase-config.js` - Firebase SDK configuration and Google OAuth helpers

### User Flow

#### **For Patients (Google Signup)**
1. Click "Continue with Google" on signup page
2. Authenticate with Google
3. Select "Patient" role
4. Account created and logged in immediately
5. Redirect to patient dashboard

#### **For Doctors (Google Signup)**
1. Click "Continue with Google" on signup page
2. Authenticate with Google
3. Select "Doctor" role
4. Redirect to verification page
5. Upload qualifications, PAN card, and provide phone number
6. Account pending admin approval
7. Admin approves → Doctor can login

#### **For Existing Users (Google Login)**
1. Click "Continue with Google" on login page
2. If email matches existing account → Link Google ID to account
3. Login successful and redirect to appropriate dashboard

## Security Considerations

### **Data Protection**
- Google profile photos downloaded and stored locally
- Google IDs stored securely in database
- Email verification status preserved from Google OAuth
- Existing security measures maintained

### **Authentication Flow**
- Google OAuth tokens not stored server-side
- Session management unchanged
- Role-based access control preserved
- Doctor approval process maintained

### **Error Handling**
- Graceful fallbacks for photo download failures
- Clear error messages for authentication failures
- Validation for required doctor verification fields

## Testing Checklist

### **Google Login Testing**
- [ ] New Google user login redirects to signup
- [ ] Existing user email matches and links Google account
- [ ] Doctor account approval status respected
- [ ] Session management works correctly
- [ ] Logout functionality preserved

### **Google Signup Testing**
- [ ] Patient signup creates account and logs in
- [ ] Doctor signup redirects to verification page
- [ ] Profile photos download correctly
- [ ] Role selection modal works
- [ ] Duplicate email handling

### **Doctor Verification Testing**
- [ ] Qualification PDF upload required
- [ ] PAN card verification (if enabled)
- [ ] Phone number validation
- [ ] Admin approval workflow
- [ ] Post-approval login access

### **Integration Testing**
- [ ] Existing username/password auth still works
- [ ] Database migrations successful
- [ ] File upload directories accessible
- [ ] Socket.IO events for new registrations
- [ ] Flash messages and error handling

## Troubleshooting

### **Common Issues**

#### **Firebase Configuration Errors**
- Verify API key and project settings in `firebase-config.js`
- Check domain authorization in Firebase Console
- Ensure Google provider is enabled

#### **Photo Download Failures**
- Check internet connectivity for photo download
- Verify `static/uploads/photos/` directory permissions
- Monitor server logs for download errors

#### **Database Issues**
- Ensure database is writable for schema migration
- Check `google_id` column was added successfully
- Verify user creation with Google data

#### **Role-based Access**
- Confirm doctor verification flow works
- Test admin approval process
- Verify role-based dashboard redirects

## Deployment Notes

### **Production Considerations**
1. Update Firebase configuration with production domain
2. Ensure HTTPS for secure Google OAuth
3. Configure proper CORS settings
4. Monitor photo storage disk usage
5. Set up error logging for OAuth failures

### **Performance Optimization**
- Photo downloads are async and don't block user creation
- Firebase SDK loaded via CDN for better caching
- Minimal JavaScript footprint for OAuth functionality

## Future Enhancements

### **Potential Improvements**
1. **Social Media Integration**: Add Facebook, LinkedIn OAuth
2. **Profile Sync**: Sync Google profile updates automatically
3. **Photo Management**: Allow users to update Google-imported photos
4. **Advanced Verification**: Use Google Workspace for doctor verification
5. **SSO Integration**: Enterprise SSO for hospital systems

### **Security Enhancements**
1. **Two-Factor Authentication**: Add 2FA for sensitive accounts
2. **OAuth Scope Management**: Limit Google permissions requested
3. **Session Security**: Enhanced session management for OAuth users
4. **Audit Logging**: Track OAuth authentication events

## Support and Maintenance

### **Regular Tasks**
- Monitor Firebase usage and quotas
- Update Firebase SDK versions
- Review OAuth security settings
- Clean up orphaned profile photos

### **Contact Information**
For technical support with this integration, refer to:
- Firebase documentation
- Flask-SocketIO documentation
- Application logs for debugging

---

## Summary

This Firebase Google OAuth integration provides a modern, secure authentication option while maintaining all existing functionality. The implementation prioritizes user experience, security, and maintainability, ensuring that Sentinel Diagnostics can serve both new Google users and existing traditional users seamlessly.