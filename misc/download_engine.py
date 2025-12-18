#!/usr/bin/env python3
"""Script to download MineStudio simulator engine non-interactively."""
import os
import sys

# Set the directory before importing minestudio
# This must be set before any minestudio imports
if 'MINESTUDIO_DIR' not in os.environ:
    os.environ['MINESTUDIO_DIR'] = '/opt/minestudio'

from minestudio.utils import get_mine_studio_dir
from minestudio.simulator.entry import download_engine

def main():
    # Verify the directory is set correctly
    mine_studio_dir = get_mine_studio_dir()
    print(f'Using MineStudio directory: {mine_studio_dir}')
    
    engine_path = os.path.join(mine_studio_dir, 'engine', 'build', 'libs', 'mcprec-6.13.jar')
    
    if os.path.exists(engine_path):
        print(f'Engine already exists at {engine_path}')
        print(f'Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB')
        return 0
    
    print(f'Engine not found at {engine_path}')
    print(f'Downloading simulator engine to {mine_studio_dir}...')
    try:
        download_engine()
        
        # Verify download succeeded
        if os.path.exists(engine_path):
            print(f'Engine downloaded successfully to {engine_path}')
            print(f'Engine size: {os.path.getsize(engine_path) / (1024*1024):.2f} MB')
            return 0
        else:
            print(f'ERROR: Engine download completed but file not found at {engine_path}', file=sys.stderr)
            return 1
    except Exception as e:
        print(f'Error downloading engine: {e}', file=sys.stderr)
        import traceback
        traceback.print_exc()
        return 1

if __name__ == '__main__':
    sys.exit(main())

