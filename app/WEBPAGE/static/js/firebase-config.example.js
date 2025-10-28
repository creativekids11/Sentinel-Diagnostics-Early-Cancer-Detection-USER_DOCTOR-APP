// Firebase Configuration Template for Sentinel Diagnostics
// Copy this file to firebase-config.js and replace with your actual Firebase configuration

// SECURITY WARNING: Never commit actual API keys to version control
// Use environment variables or secure configuration management

const firebaseConfig = {
    apiKey: "your_firebase_api_key_here",
    authDomain: "your-project.firebaseapp.com", 
    projectId: "your-project-id",
    storageBucket: "your-project.firebasestorage.app",
    messagingSenderId: "your_sender_id",
    appId: "your_app_id"
};

// Initialize Firebase
import { initializeApp } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-app.js';
import { getAuth, GoogleAuthProvider, signInWithPopup, signOut } from 'https://www.gstatic.com/firebasejs/10.7.1/firebase-auth.js';

const app = initializeApp(firebaseConfig);
const auth = getAuth(app);
const provider = new GoogleAuthProvider();

// Configure Google provider
provider.addScope('profile');
provider.addScope('email');

// Google Sign In function
export async function signInWithGoogle() {
    try {
        const result = await signInWithPopup(auth, provider);
        const user = result.user;
        
        // Extract user information
        const userData = {
            uid: user.uid,
            email: user.email,
            displayName: user.displayName,
            photoURL: user.photoURL,
            emailVerified: user.emailVerified
        };
        
        return {
            success: true,
            user: userData,
            credential: GoogleAuthProvider.credentialFromResult(result)
        };
    } catch (error) {
        console.error('Google sign-in error:', error);
        return {
            success: false,
            error: error.message,
            errorCode: error.code
        };
    }
}

// Sign out function
export async function signOutUser() {
    try {
        await signOut(auth);
        return { success: true };
    } catch (error) {
        console.error('Sign out error:', error);
        return { success: false, error: error.message };
    }
}

// Listen for auth state changes
export function onAuthStateChanged(callback) {
    return auth.onAuthStateChanged(callback);
}

// Get current user
export function getCurrentUser() {
    return auth.currentUser;
}