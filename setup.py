#!/usr/bin/env python3
"""
Setup and Run Script for RAG Document QA Chatbot
This script will help you set up and run the chatbot system
"""

import subprocess
import sys
import os
import webbrowser
import time
import signal
from pathlib import Path


def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed!")
        print(f"Error: {e.stderr}")
        return False


def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major == 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} is not compatible")
        print("Please use Python 3.8 or higher")
        return False


def install_requirements():
    """Install required packages"""
    requirements_file = Path("requirements.txt")
    if not requirements_file.exists():
        print("❌ requirements.txt not found!")
        return False

    return run_command(
        f"{sys.executable} -m pip install -r requirements.txt",
        "Installing requirements"
    )


def create_directories():
    """Create necessary directories"""
    print("\n📁 Creating directories...")
    directories = ["uploads", "vector_db"]
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        print(f"✅ Created/verified directory: {directory}")
    return True


def download_nltk_data():
    """Download required NLTK data"""
    print("\n📚 Downloading NLTK data...")
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        print("✅ NLTK punkt data downloaded")
        return True
    except Exception as e:
        print(f"❌ Failed to download NLTK data: {e}")
        return False


def check_env_file():
    """Check if .env file exists and has required variables"""
    env_file = Path(".env")
    if not env_file.exists():
        print("⚠️ .env file not found!")
        print("Creating a basic .env file...")

        env_content = """# API Configuration
GEMINI_API_KEY=AIzaSyADnVU3f1GSJ55KUfTd26bQTUcGZQ__NKA
VALID_TOKEN=5b6105937b7cc769e46557d6241353e800d99cb57def59fd962d1d6ea8fcf736

# RAG Configuration
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
CHUNK_SIZE=1000
CHUNK_OVERLAP=200
TOP_K_RETRIEVAL=5
MAX_CONTEXT_LENGTH=4000

# Environment
ENVIRONMENT=development
DEBUG=true
"""
        with open(env_file, 'w') as f:
            f.write(env_content)
        print("✅ Created .env file with default values")
    else:
        print("✅ .env file exists")

    return True


def start_server():
    """Start the FastAPI server"""
    print("\n🚀 Starting FastAPI server...")
    print("Server will start at: http://localhost:8000")
    print("API documentation: http://localhost:8000/api/v1/docs")
    print("\nPress Ctrl+C to stop the server")

    try:
        # Start the server
        process = subprocess.Popen([
            sys.executable, "main.py"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # Wait a moment for server to start
        time.sleep(3)

        # Check if server is running
        try:
            import requests
            response = requests.get("http://localhost:8000/api/v1/health", timeout=5)
            if response.status_code == 200:
                print("✅ Server is running successfully!")

                # Open the chatbot interface
                chatbot_url = "file://" + str(Path("chatbot.html").absolute())
                print(f"\n🌐 Opening chatbot interface: {chatbot_url}")
                webbrowser.open(chatbot_url)

            else:
                print("⚠️ Server started but health check failed")
        except Exception as e:
            print(f"⚠️ Could not verify server status: {e}")

        # Wait for user to stop the server
        try:
            process.wait()
        except KeyboardInterrupt:
            print("\n🛑 Stopping server...")
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
            print("✅ Server stopped")

    except Exception as e:
        print(f"❌ Failed to start server: {e}")
        return False

    return True


def create_chatbot_html():
    """Create the chatbot HTML file if it doesn't exist"""
    html_file = Path("chatbot.html")
    if html_file.exists():
        print("✅ chatbot.html already exists")
        return True

    print("📝 Creating chatbot.html...")
    # This would contain the HTML content - for now, just check if it exists
    if not html_file.exists():
        print("❌ chatbot.html not found!")
        print("Please ensure the chatbot.html file is in the current directory")
        return False

    return True


def main():
    """Main setup and run function"""
    print("=" * 60)
    print("🤖 RAG Document QA Chatbot - Setup & Run")
    print("=" * 60)

    # Check Python version
    if not check_python_version():
        sys.exit(1)

    # Check environment file
    if not check_env_file():
        sys.exit(1)

    # Create directories
    if not create_directories():
        sys.exit(1)

    # Install requirements
    print("\n" + "=" * 40)
    install_choice = input("Install/update requirements? (y/N): ").lower().strip()
    if install_choice in ['y', 'yes']:
        if not install_requirements():
            print("⚠️ Requirements installation failed, but continuing...")

    # Download NLTK data
    if not download_nltk_data():
        print("⚠️ NLTK data download failed, but continuing...")

    # Check for chatbot HTML
    if not create_chatbot_html():
        sys.exit(1)

    print("\n" + "=" * 40)
    print("🎉 Setup completed!")
    print("\n📋 Quick Start Guide:")
    print("1. Server will start automatically")
    print("2. Chatbot interface will open in your browser")
    print("3. Click 'Test Connection' in the chatbot")
    print("4. Upload a document or enter a URL")
    print("5. Start asking questions!")

    print("\n🔧 Troubleshooting:")
    print("- If connection fails, check server logs")
    print("- Make sure no other service is using port 8000")
    print("- Check firewall settings if needed")

    start_choice = input("\nStart the server now? (Y/n): ").lower().strip()
    if start_choice not in ['n', 'no']:
        start_server()
    else:
        print("\n📝 To start manually later:")
        print("1. Run: python main.py")
        print("2. Open: chatbot.html in your browser")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n👋 Setup cancelled by user")
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)