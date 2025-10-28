// Firebase Configuration for Sentinel Diagnostics
// This file contains Firebase SDK initialization and Google OAuth setup

// Firebase configuration (replace with your actual config)
const firebaseConfig = {
    apiKey: "AIzaSyDW0GFqn8-VUX0vn28OSqIsZeYtzi1AT7o",
    authDomain: "login-system-69d80.firebaseapp.com",
    projectId: "login-system-69d80",
    storageBucket: "login-system-69d80.firebasestorage.app",
    messagingSenderId: "118348051351",
    appId: "1:118348051351:web:081b0aefad698042558ad2"
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