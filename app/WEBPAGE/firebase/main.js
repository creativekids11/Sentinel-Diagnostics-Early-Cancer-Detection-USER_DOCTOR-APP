import { initializeApp } from "https://www.gstatic.com/firebasejs/12.4.0/firebase-app.js";
import { getAuth ,GoogleAuthProvider , signInWithPopup} from "https://www.gstatic.com/firebasejs/12.4.0/firebase-auth.js";

  const firebaseConfig = {
    apiKey: "AIzaSyDW0GFqn8-VUX0vn28OSqIsZeYtzi1AT7o",
    authDomain: "login-system-69d80.firebaseapp.com",
    projectId: "login-system-69d80",
    storageBucket: "login-system-69d80.firebasestorage.app",
    messagingSenderId: "118348051351",
    appId: "1:118348051351:web:081b0aefad698042558ad2"
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