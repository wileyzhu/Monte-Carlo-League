#!/usr/bin/env python3
"""
Setup script for Worlds 2025 Tournament Simulator Web App
"""

import subprocess
import sys
import os

def install_requirements():
    """Install required packages"""
    print("📦 Installing required packages...")
    try:
        subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'])
        print("✅ Requirements installed successfully!")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ Error installing requirements: {e}")
        return False

def check_data_files():
    """Check if required data files exist"""
    print("📊 Checking data files...")
    
    required_files = [
        'dataset/probability_matrix_msi_adjusted.csv',
        'dataset/probability_matrix.csv'
    ]
    
    missing_files = []
    for file_path in required_files:
        if not os.path.exists(file_path):
            missing_files.append(file_path)
    
    if missing_files:
        print("⚠️  Missing data files:")
        for file_path in missing_files:
            print(f"   - {file_path}")
        print("\n🔧 Generating missing data files...")
        
        try:
            # Generate probability matrix if missing
            if 'probability_matrix.csv' in str(missing_files):
                print("   Generating base probability matrix...")
                subprocess.check_call([sys.executable, 'probability_matrix.py'])
            
            # Generate MSI-adjusted matrix if missing
            if 'probability_matrix_msi_adjusted.csv' in str(missing_files):
                print("   Applying MSI-based regional adjustments...")
                subprocess.check_call([sys.executable, 'regional_adjustment.py'])
            
            print("✅ Data files generated successfully!")
            return True
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error generating data files: {e}")
            return False
    else:
        print("✅ All required data files found!")
        return True

def main():
    """Main setup function"""
    print("🏆 Setting up Worlds 2025 Tournament Simulator Web App")
    print("=" * 60)
    
    # Install requirements
    if not install_requirements():
        print("❌ Setup failed during package installation")
        sys.exit(1)
    
    # Check data files
    if not check_data_files():
        print("❌ Setup failed during data file check")
        sys.exit(1)
    
    print("\n" + "=" * 60)
    print("🎉 Setup completed successfully!")
    print("\n🚀 To start the web app, run:")
    print("   python run_webapp.py")
    print("\n📱 Then open your browser to: http://localhost:5000")
    print("=" * 60)

if __name__ == '__main__':
    main()