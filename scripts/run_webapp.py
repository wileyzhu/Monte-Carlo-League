#!/usr/bin/env python3
"""
Worlds 2025 Tournament Simulator Web App
Startup script for the Flask web application
"""

import os
import sys
from app import app

def main():
    """Main function to start the web app"""
    print("🏆 Starting Worlds 2025 Tournament Simulator Web App...")
    print("📊 Loading Bayesian AR(3) models and MSI 2025 data...")
    
    # Set environment variables
    os.environ['FLASK_ENV'] = 'development'
    
    try:
        # Start the Flask app
        print("🚀 Web app starting on http://localhost:5003")
        print("📱 Open your browser and navigate to the URL above")
        print("⏹️  Press Ctrl+C to stop the server")
        print("-" * 50)
        
        app.run(
            debug=True,
            host='0.0.0.0',
            port=5003,
            use_reloader=True
        )
        
    except KeyboardInterrupt:
        print("\n👋 Shutting down the web app...")
        sys.exit(0)
    except Exception as e:
        print(f"❌ Error starting web app: {e}")
        sys.exit(1)

if __name__ == '__main__':
    main()