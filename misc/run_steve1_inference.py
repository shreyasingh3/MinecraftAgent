#!/usr/bin/env python3
"""
Minimal STEVE-1 inference script for MineStudio.
This script demonstrates how to load STEVE-1 and run inference on a simple task.
"""

import os
import sys
import torch
import numpy as np
import gc
from pathlib import Path

# Set auto-download flag to prevent interactive prompts
os.environ['AUTO_DOWNLOAD_ENGINE'] = '1'

from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import RecordCallback, CommandsCallback, JudgeResetCallback
from minestudio.models.steve_one import SteveOnePolicy

def main():
    print("=" * 60)
    print("STEVE-1 Inference Test")
    print("=" * 60)
    
    # Detect device
    if torch.cuda.is_available():
        device = "cuda"
        print("Using CUDA device")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
        print("Using MPS device (Apple Silicon)")
    else:
        device = "cpu"
        print("Using CPU device")
    
    # Load STEVE-1 model
    print("\nLoading STEVE-1 model from HuggingFace...")
    print("This will download ~2GB on first run. Please wait...")
    try:
        model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path("./output/steve1_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Setup environment with a simple task: collect wood
    print("\nSetting up Minecraft environment...")
    task_text = "collect wood"  # Natural language instruction for STEVE-1
    
    env = MinecraftSim(
        obs_size=(128, 128),
        max_mem=os.environ.get("MINECRAFT_MAX_MEM", "512M"),
        callbacks=[
            CommandsCallback(commands=[
                # Give the agent some basic tools
                '/give @p minecraft:iron_axe 1',
            ]),
            JudgeResetCallback(time_limit=600),  # Reset if stuck
            RecordCallback(
                record_path=str(output_dir),
                fps=30,
                frame_type="pov"
            ),
        ]
    )
    
    try:
        # Prepare model condition
        print("\nPreparing model condition...")
        condition = model.prepare_condition({
            'cond_scale': 4.0,
            'text': task_text
        })
        state_in = model.initial_state(condition, 1)
        print("✓ Condition prepared")
        
        # Reset environment
        print("\nResetting environment...")
        obs, info = env.reset()
        print("✓ Environment ready")
        
        # Run inference
        print(f"\nRunning inference for task: '{task_text}'")
        print("This will run for up to 600 steps...")
        print("-" * 60)
        
        for step in range(600):
            try:
                # Get action from STEVE-1
                action, state_in = model.get_steve_action(
                    condition,
                    obs,
                    state_in,
                    input_shape='*',
                    deterministic=False
                )
                
                # Convert action to numpy if needed
                if isinstance(action, dict):
                    action_numpy = {}
                    for k, v in action.items():
                        if isinstance(v, torch.Tensor):
                            action_numpy[k] = v.cpu().numpy()
                        else:
                            action_numpy[k] = v
                    action = action_numpy
                
                # Step environment
                obs, reward, terminated, truncated, info = env.step(action)
                
                # Progress update
                if (step + 1) % 50 == 0:
                    print(f"Step {step + 1}/600")
                
                # Check for early termination
                if terminated or truncated:
                    print(f"\n✓ Episode completed at step {step + 1}")
                    break
                    
            except KeyboardInterrupt:
                print("\n\nInterrupted by user")
                break
            except Exception as e:
                print(f"\n✗ Error at step {step}: {e}")
                import traceback
                traceback.print_exc()
                break
        
        print("\n" + "=" * 60)
        print("Inference completed!")
        print(f"Video saved to: {output_dir}/episode_1.mp4")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ Error during execution: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
        
    finally:
        # Cleanup
        env.close()
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print("\n✓ Cleanup complete")

if __name__ == "__main__":
    main()

