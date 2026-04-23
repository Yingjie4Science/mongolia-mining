"""
Setup script to install dependencies and run LULC analysis
Run this first to ensure all packages are installed
"""

import subprocess
import sys

def install_packages():
    """Install required packages using pip."""
    packages = [
        'numpy',
        'pandas',
        'matplotlib',
        'seaborn',
        'rasterio',
        'geopandas',
        'shapely'
    ]
    
    print("\n" + "="*70)
    print("INSTALLING REQUIRED PACKAGES")
    print("="*70)
    
    for package in packages:
        print(f"\n[Installing] {package}...", end=" ")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", package])
            print("✓ Done")
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to install {package}")
            print(f"Error: {e}")
    
    print("\n" + "="*70)
    print("INSTALLATION COMPLETE")
    print("="*70)

def verify_packages():
    """Verify all packages are importable."""
    print("\n[Verifying] Checking package imports...")
    
    packages = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'matplotlib': 'Matplotlib',
        'seaborn': 'Seaborn',
        'rasterio': 'Rasterio',
        'geopandas': 'GeoPandas',
        'shapely': 'Shapely'
    }
    
    all_good = True
    for pkg, name in packages.items():
        try:
            __import__(pkg)
            print(f"  ✓ {name}")
        except ImportError:
            print(f"  ✗ {name} - MISSING")
            all_good = False
    
    if all_good:
        print("\n✓ All packages verified successfully!")
        return True
    else:
        print("\n✗ Some packages are still missing. Please install manually:")
        print("  conda install -c conda-forge numpy pandas matplotlib seaborn rasterio geopandas shapely")
        return False

if __name__ == "__main__":
    print("\n" + "="*70)
    print("LULC CHANGE ANALYSIS - SETUP SCRIPT")
    print("="*70)
    
    install_packages()
    
    if verify_packages():
        print("\n" + "="*70)
        print("✓ Ready to run LULC analysis!")
        print("="*70)
        print("\nNext steps:")
        print("1. Edit 'lulc_change_analysis.py'")
        print("2. Update the 'dir' variable to point to your data folder")
        print("3. Run: python lulc_change_analysis.py")
    else:
        print("\nTroubleshooting:")
        print("If conda is available, try:")
        print("  conda install -c conda-forge rasterio geopandas")
