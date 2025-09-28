#!/usr/bin/env python3
"""
Digital Twin Wellness Optimizer - Streamlit App Runner
Run this script to launch the interactive digital twin visualization app.
"""

import subprocess
import sys
import os

def main():
    """Launch the Streamlit app."""
    # Get the directory of this script
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "streamlit_app.py")

    # Path to virtual environment python
    venv_python = os.path.join(script_dir, "..", "venv", "bin", "python3")

    if not os.path.exists(app_path):
        print("Error: streamlit_app.py not found in the current directory.")
        sys.exit(1)

    if not os.path.exists(venv_python):
        print("Error: Virtual environment python not found. Please ensure venv is set up.")
        sys.exit(1)

    print("🚀 Launching Digital Twin Wellness Optimizer...")
    print("📱 Opening interactive visualization app...")
    print("💡 Use Ctrl+C to stop the server")

    try:
        # Set up environment with correct Python path
        env = os.environ.copy()
        # Add the parent directory to PYTHONPATH so digital_twin_framework can be imported
        parent_dir = os.path.dirname(script_dir)
        env['PYTHONPATH'] = parent_dir + os.pathsep + env.get('PYTHONPATH', '')

        # Run streamlit with the app using venv python
        subprocess.run([venv_python, "-m", "streamlit", "run", app_path],
                      env=env, check=True)
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error running Streamlit app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()