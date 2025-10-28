#!/usr/bin/env python3
"""
Enhanced Production WSGI for AI Medical Diagnosis Platform
Handles import errors gracefully and provides detailed diagnostics
"""

import os
import sys
import logging
import traceback
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('app.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def setup_python_paths():
    """Setup Python paths for proper imports"""
    # Get absolute paths
    current_dir = Path(__file__).parent.absolute()
    app_dir = current_dir / 'app'
    webpage_dir = app_dir / 'WEBPAGE'
    
    # Add paths to sys.path
    paths_to_add = [str(current_dir), str(app_dir), str(webpage_dir)]
    
    for path in paths_to_add:
        if path not in sys.path:
            sys.path.insert(0, path)
    
    logger.info(f"Repository root: {current_dir}")
    logger.info(f"App directory: {app_dir}")
    logger.info(f"WEBPAGE directory: {webpage_dir}")
    logger.info(f"Python paths added: {paths_to_add}")
    
    return current_dir, app_dir, webpage_dir

def check_dependencies():
    """Check if critical dependencies are available"""
    critical_packages = [
        'flask', 'numpy', 'pandas', 'torch', 'cv2', 'PIL'
    ]
    
    missing_packages = []
    available_packages = []
    
    for package in critical_packages:
        try:
            if package == 'cv2':
                import cv2
                available_packages.append(f"opencv-python ({cv2.__version__})")
            elif package == 'PIL':
                import PIL
                available_packages.append(f"Pillow ({PIL.__version__})")
            else:
                module = __import__(package)
                version = getattr(module, '__version__', 'unknown')
                available_packages.append(f"{package} ({version})")
        except ImportError:
            missing_packages.append(package)
    
    logger.info(f"Available packages: {', '.join(available_packages)}")
    if missing_packages:
        logger.warning(f"Missing packages: {', '.join(missing_packages)}")
    
    return len(missing_packages) == 0

def create_fallback_app():
    """Create a fallback Flask app with diagnostics"""
    from flask import Flask
    
    fallback_app = Flask(__name__)
    
    @fallback_app.route('/')
    def index():
        return f"""
        <html>
        <head>
            <title>AI Medical Platform - Loading</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
                .container {{ max-width: 800px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                .status {{ padding: 15px; margin: 10px 0; border-radius: 5px; }}
                .error {{ background: #fee; border: 1px solid #fcc; color: #c33; }}
                .warning {{ background: #fff3cd; border: 1px solid #ffeaa7; color: #856404; }}
                .info {{ background: #e3f2fd; border: 1px solid #bbdefb; color: #1565c0; }}
                .loading {{ background: #f0f8ff; border: 1px solid #87ceeb; color: #4682b4; }}
                code {{ background: #f8f9fa; padding: 2px 4px; border-radius: 3px; }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>🏥 AI Medical Diagnosis Platform</h1>
                
                <div class="loading">
                    <h3>⏳ Platform Status: Initializing</h3>
                    <p>The AI Medical Platform is currently loading. This may take 1-2 minutes.</p>
                </div>
                
                <h3>🔧 System Information</h3>
                <div class="info">
                    <p><strong>Python Version:</strong> {sys.version}</p>
                    <p><strong>Working Directory:</strong> <code>{os.getcwd()}</code></p>
                    <p><strong>Repository Root:</strong> <code>{Path(__file__).parent.absolute()}</code></p>
                </div>
                
                <h3>📦 Package Status</h3>
                <div class="info">
                    <p>Checking critical dependencies...</p>
                    <ul>
                        <li>Flask: {'✓' if check_package_available('flask') else '✗'}</li>
                        <li>NumPy: {'✓' if check_package_available('numpy') else '✗'}</li>
                        <li>Pandas: {'✓' if check_package_available('pandas') else '✗'}</li>
                        <li>PyTorch: {'✓' if check_package_available('torch') else '✗'}</li>
                        <li>OpenCV: {'✓' if check_package_available('cv2') else '✗'}</li>
                        <li>Pillow: {'✓' if check_package_available('PIL') else '✗'}</li>
                    </ul>
                </div>
                
                <h3>🔄 Next Steps</h3>
                <div class="warning">
                    <p>If the main application doesn't load within 2 minutes:</p>
                    <ol>
                        <li>Check the console/terminal for error messages</li>
                        <li>Review the <code>app.log</code> file for detailed logs</li>
                        <li>Ensure all dependencies are installed correctly</li>
                        <li>Try refreshing this page</li>
                    </ol>
                </div>
                
                <div class="info">
                    <p><strong>Expected Features:</strong></p>
                    <ul>
                        <li>🧠 Brain Tumor Detection & Segmentation</li>
                        <li>🫁 Lung Cancer Detection with YOLO</li>
                        <li>🎗️ Breast Cancer Analysis with MLMO-EMO</li>
                        <li>🤖 AI Assistant Chat Interface</li>
                        <li>📊 Medical Report Generation</li>
                        <li>👥 Doctor/Patient Management</li>
                    </ul>
                </div>
            </div>
            
            <script>
                // Auto-refresh every 30 seconds
                setTimeout(function(){{ window.location.reload(); }}, 30000);
            </script>
        </body>
        </html>
        """
    
    @fallback_app.route('/health')
    def health():
        return {{"status": "fallback", "message": "Main app not yet loaded"}}
    
    return fallback_app

def check_package_available(package_name):
    """Check if a package is available for import"""
    try:
        if package_name == 'cv2':
            import cv2
        elif package_name == 'PIL':
            import PIL
        else:
            __import__(package_name)
        return True
    except ImportError:
        return False

def main():
    """Main application loader"""
    logger.info("Starting AI Medical Diagnosis Platform...")
    
    # Setup paths
    current_dir, app_dir, webpage_dir = setup_python_paths()
    
    # Check dependencies
    deps_ok = check_dependencies()
    
    try:
        # Try to import the main application
        logger.info("Attempting to import main application...")
        
        # Change to webpage directory for proper imports
        os.chdir(str(webpage_dir))
        
        # Ensure utils directory is in Python path
        utils_path = os.path.join(str(webpage_dir), 'utils')
        if utils_path not in sys.path:
            sys.path.insert(0, utils_path)
        
        logger.info(f"Current working directory: {os.getcwd()}")
        logger.info(f"Python path includes: {[p for p in sys.path if 'deployed-hackathon' in p]}")
        
        # Import main app
        from app import app, socketio
        
        # Initialize AI models
        logger.info("Initializing AI models...")
        from app import initialize_models
        initialize_models()
        
        logger.info("✓ AI Medical Diagnosis Platform loaded successfully")
        
        # Configure the app
        app.config['SECRET_KEY'] = os.environ.get('FLASK_SECRET', 'dev-secret-change-in-production')
        
        return app, socketio
        
    except ImportError as e:
        logger.error(f"Import error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return fallback app
        logger.info("Using fallback diagnostic app")
        return create_fallback_app(), None
        
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(f"Full traceback: {traceback.format_exc()}")
        
        # Return fallback app
        return create_fallback_app(), None

# Initialize the application
app, socketio = main()

if __name__ == "__main__":
    logger.info("Starting development server...")
    if socketio:
        socketio.run(app, host='0.0.0.0', port=5000, debug=False)
    else:
        app.run(host='0.0.0.0', port=5000, debug=False)
else:
    # WSGI mode
    logger.info("Running in WSGI production mode")
    application = app