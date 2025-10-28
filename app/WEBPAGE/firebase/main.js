import { initializeApp } from "https://www.gstatic.com/firebasejs/12.4.0/firebase-app.js";
import { getAuth ,GoogleAuthProvider , signInWithPopup} from "https://www.gstatic.com/firebasejs/12.4.0/firebase-auth.js";

  // SECURITY WARNING: Replace with your actual Firebase configuration
  // Never commit real API keys to version control
  const firebaseConfig = {
    apiKey: window.FIREBASE_API_KEY || "your_firebase_api_key_here",
    authDomain: "your-project.firebaseapp.com",
    projectId: "your-project-id", 
    storageBucket: "your-project.firebasestorage.app",
    messagingSenderId: "your_sender_id",
    appId: "your_app_id"
  };


  const app = initializeApp(firebaseConfig);
  const auth = getAuth(app);
  auth.languageCode = 'en';
  const provider = new GoogleAuthProvider();

document.getElementById('google-signin-btn').addEventListener('click', async (e) => {
  e.preventDefault();
  try {
    const result = await signInWithPopup(auth, provider);
    console.log('User signed in:', result.user);
    alert('Signed in as ' + result.user.displayName);
    // Optionally redirect: window.location.href = '/dashboard.html';
  } catch (error) {
    console.error('Error during sign-in:', error);
    alert('Sign-in failed: ' + error.message);
  }
});