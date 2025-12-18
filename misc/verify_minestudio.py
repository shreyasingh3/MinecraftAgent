#!/usr/bin/env python3
"""
Verification script for MineStudio installation.
Tests that all components are properly installed and accessible.
"""

import os
import sys

# Set auto-download flag
os.environ['AUTO_DOWNLOAD_ENGINE'] = '1'

def check_imports():
    """Check that all required packages can be imported"""
    print("Checking Python package imports...")
    
    packages = [
        ('torch', 'PyTorch'),
        ('numpy', 'NumPy'),
        ('cv2', 'OpenCV'),
        ('minestudio', 'MineStudio'),
    ]
    
    all_ok = True
    for module_name, display_name in packages:
        try:
            __import__(module_name)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    return all_ok

def check_java():
    """Check that Java is installed and accessible"""
    print("\nChecking Java installation...")
    
    import subprocess
    try:
        result = subprocess.run(
            ['java', '-version'],
            capture_output=True,
            text=True
        )
        # Java -version outputs to stderr, not stdout, and returns non-zero
        output = result.stderr if result.stderr else result.stdout
        if output:  # Java -version always outputs to stderr
            version_line = output.split('\n')[0] if output else "Java found"
            print(f"  ✓ Java found: {version_line}")
            
            # Check if it's Java 8
            if '1.8' in version_line or 'openjdk version "8' in version_line or 'version "8' in version_line:
                print("  ✓ Java version 8 detected (correct)")
            else:
                print("  ⚠ Warning: Java 8 recommended, but other versions may work")
            return True
        else:
            print("  ✗ Java not found or not working")
            return False
    except FileNotFoundError:
        print("  ✗ Java command not found")
        print("    Install with: conda install --channel=conda-forge openjdk=8 -y")
        return False

def check_torch_device():
    """Check PyTorch device availability"""
    print("\nChecking PyTorch device support...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        
        if torch.cuda.is_available():
            print(f"  ✓ CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print("  - CUDA not available")
        
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("  ✓ MPS available (Apple Silicon)")
        else:
            print("  - MPS not available")
        
        print("  ✓ CPU always available")
        return True
    except Exception as e:
        print(f"  ✗ Error checking PyTorch: {e}")
        return False

def check_minestudio_components():
    """Check that MineStudio components can be imported"""
    print("\nChecking MineStudio components...")
    
    components = [
        ('minestudio.simulator', 'MinecraftSim'),
        ('minestudio.models.steve_one', 'SteveOnePolicy'),
        ('minestudio.simulator.callbacks', 'Callbacks'),
    ]
    
    all_ok = True
    for module_path, display_name in components:
        try:
            __import__(module_path)
            print(f"  ✓ {display_name}")
        except ImportError as e:
            print(f"  ✗ {display_name}: {e}")
            all_ok = False
    
    return all_ok

def check_engine():
    """Check if simulator engine is available"""
    print("\nChecking simulator engine...")
    
    try:
        from minestudio.utils import get_mine_studio_dir
        mine_studio_dir = get_mine_studio_dir()
        engine_path = os.path.join(mine_studio_dir, "engine", "build", "libs", "mcprec-6.13.jar")
        
        if os.path.exists(engine_path):
            print(f"  ✓ Engine found at: {engine_path}")
            return True
        else:
            print(f"  - Engine not found (will auto-download on first use)")
            print(f"    Expected location: {engine_path}")
            return True  # Not an error, will download automatically
    except Exception as e:
        print(f"  ✗ Error checking engine: {e}")
        return False

def main():
    print("=" * 60)
    print("MineStudio Installation Verification")
    print("=" * 60)
    
    results = []
    
    results.append(("Python packages", check_imports()))
    results.append(("Java installation", check_java()))
    results.append(("PyTorch devices", check_torch_device()))
    results.append(("MineStudio components", check_minestudio_components()))
    results.append(("Simulator engine", check_engine()))
    
    # Summary
    print("\n" + "=" * 60)
    print("Verification Summary")
    print("=" * 60)
    
    all_passed = True
    for name, passed in results:
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"{status}: {name}")
        if not passed:
            all_passed = False
    
    print("=" * 60)
    
    if all_passed:
        print("\n✓ All checks passed! MineStudio is ready to use.")
        print("\nNext steps:")
        print("  1. Run: python3 run_steve1_inference.py")
        print("  2. Run: python3 run_agentbeats_evaluation.py --task collect_wood --difficulty simple")
        return 0
    else:
        print("\n✗ Some checks failed. Please fix the issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())

