"""
Setup and Run Script for SDG 14 Fish Conservation Project
This script automates the entire setup process
"""

import subprocess
import sys
import os

def print_header(message):
    print("\n" + "="*70)
    print(f"  {message}")
    print("="*70 + "\n")

def run_command(description, command):
    print(f"→ {description}...")
    try:
        subprocess.run(command, check=True, shell=True)
        print(f"✓ {description} completed successfully!\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Error during {description}: {e}\n")
        return False

def main():
    print_header("SDG 14 Fish Conservation Status Prediction - Setup")
    
    print("This script will:")
    print("1. Check/install required packages")
    print("2. Generate the fish conservation dataset")
    print("3. Train and compare ML models")
    print("4. Launch the Streamlit web application")
    
    input("\nPress Enter to continue...")
    
    # Step 1: Check Python version
    print_header("Step 1: Checking Python Version")
    python_version = sys.version_info
    print(f"Python version: {python_version.major}.{python_version.minor}.{python_version.micro}")
    
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 8):
        print("✗ Python 3.8 or higher is required!")
        return
    print("✓ Python version is compatible\n")
    
    # Step 2: Install requirements
    print_header("Step 2: Installing Required Packages")
    if not run_command("Installing packages", f"{sys.executable} -m pip install -r requirements.txt"):
        return
    
    # Step 3: Generate dataset
    print_header("Step 3: Generating Fish Conservation Dataset")
    if not run_command("Generating dataset", f"{sys.executable} generate_dataset.py"):
        return
    
    # Step 4: Train models
    print_header("Step 4: Training Machine Learning Models")
    print("This may take a few minutes...\n")
    if not run_command("Training models", f"{sys.executable} train_models.py"):
        return
    
    # Step 5: Launch Streamlit app
    print_header("Step 5: Launching Streamlit Web Application")
    print("The application will open in your default web browser.")
    print("Press Ctrl+C to stop the server.\n")
    
    try:
        subprocess.run(f"{sys.executable} -m streamlit run app.py", shell=True)
    except KeyboardInterrupt:
        print("\n\n✓ Application stopped. Thank you for using the SDG 14 Fish Conservation Predictor!")

if __name__ == "__main__":
    main()

