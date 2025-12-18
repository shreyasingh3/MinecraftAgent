#!/usr/bin/env python3
"""
Check if the Minecraft engine JAR contains macOS native libraries.
"""
import os
import sys
import subprocess
import zipfile
from pathlib import Path

def find_jar():
    """Find the Minecraft engine JAR file."""
    # Try multiple methods to find the JAR
    possible_paths = [
        # Standard MineStudio location
        os.path.expanduser("~/.minestudio/engine/build/libs/mcprec-6.13.jar"),
        # Alternative locations
        "/opt/minestudio/engine/build/libs/mcprec-6.13.jar",
        # Current directory
        "./minestudio/simulator/minerl/env/../MCP-Reborn/build/libs/mcprec-6.13.jar",
    ]
    
    # Try using minestudio utils if available
    try:
        from minestudio.utils import get_mine_studio_dir
        mine_studio_dir = get_mine_studio_dir()
        jar_path = os.path.join(mine_studio_dir, "engine", "build", "libs", "mcprec-6.13.jar")
        possible_paths.insert(0, jar_path)
    except ImportError:
        pass  # minestudio not installed, use other paths
    except Exception:
        pass  # Error getting directory, use other paths
    
    # Check each possible path
    for jar_path in possible_paths:
        if os.path.exists(jar_path):
            return jar_path
    
    return None

def check_jar_contents(jar_path):
    """Check what native libraries are in the JAR."""
    if not os.path.exists(jar_path):
        print(f"JAR not found at: {jar_path}")
        return False
    
    print(f"Checking JAR: {jar_path}")
    print(f"JAR size: {os.path.getsize(jar_path) / (1024*1024):.2f} MB")
    print()
    
    try:
        with zipfile.ZipFile(jar_path, 'r') as jar:
            files = jar.namelist()
            
            # Look for native libraries
            dylib_files = [f for f in files if f.endswith('.dylib')]
            so_files = [f for f in files if f.endswith('.so')]
            dll_files = [f for f in files if f.endswith('.dll')]
            
            # Look for LWJGL
            lwjgl_files = [f for f in files if 'lwjgl' in f.lower()]
            
            print("Native Libraries Found:")
            print(f"  macOS (.dylib): {len(dylib_files)} files")
            if dylib_files:
                print("    Examples:")
                for f in dylib_files[:5]:
                    print(f"      - {f}")
            
            print(f"  Linux (.so): {len(so_files)} files")
            if so_files:
                print("    Examples:")
                for f in so_files[:5]:
                    print(f"      - {f}")
            
            print(f"  Windows (.dll): {len(dll_files)} files")
            print()
            
            print("LWJGL-related files:")
            if lwjgl_files:
                print(f"  Found {len(lwjgl_files)} LWJGL files")
                for f in lwjgl_files[:10]:
                    print(f"    - {f}")
            else:
                print("  No LWJGL files found")
            
            print()
            
            # Check architecture
            arch_files = {}
            for f in dylib_files + so_files:
                if 'arm64' in f or 'aarch64' in f:
                    arch_files.setdefault('ARM64', []).append(f)
                elif 'x86_64' in f or 'amd64' in f:
                    arch_files.setdefault('x86_64', []).append(f)
                elif 'x86' in f or 'i386' in f:
                    arch_files.setdefault('x86', []).append(f)
            
            if arch_files:
                print("Architecture-specific files:")
                for arch, files in arch_files.items():
                    print(f"  {arch}: {len(files)} files")
            
            # Summary
            print()
            print("=" * 60)
            if dylib_files:
                print("✓ JAR contains macOS native libraries (.dylib)")
                print("  LWJGL should be able to extract and use them.")
                return True
            else:
                print("✗ JAR does NOT contain macOS native libraries (.dylib)")
                print("  This JAR was likely built for Linux only.")
                print("  Solutions:")
                print("  1. Use Docker (Linux environment)")
                print("  2. Use Rosetta 2 to run x86_64 Java (if JAR has x86_64 libs)")
                print("  3. Rebuild engine for macOS ARM64")
                return False
                
    except Exception as e:
        print(f"Error reading JAR: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    jar_path = find_jar()
    if not jar_path:
        print("Could not locate Minecraft engine JAR.")
        print("Make sure MineStudio is installed and engine is downloaded.")
        return 1
    
    has_macos_libs = check_jar_contents(jar_path)
    return 0 if has_macos_libs else 1

if __name__ == '__main__':
    sys.exit(main())

