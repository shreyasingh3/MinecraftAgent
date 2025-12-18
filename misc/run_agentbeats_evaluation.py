#!/usr/bin/env python3
"""
AgentBeats/MCU Evaluation Script with STEVE-1
This script integrates STEVE-1 with the MCU benchmark evaluation flow.
"""

import argparse
import os
import sys
import torch
import numpy as np
import gc
from pathlib import Path

# Set auto-download flag
os.environ['AUTO_DOWNLOAD_ENGINE'] = '1'

# Monkey-patch check_engine for non-interactive use
from minestudio.simulator import entry as minestudio_entry
from minestudio.utils import get_mine_studio_dir

_engine_checked = False
_engine_exists = None

def patched_check_engine():
    """Patched version that auto-downloads in non-interactive environments"""
    global _engine_checked, _engine_exists
    
    if _engine_checked and _engine_exists:
        return
    
    mine_studio_dir = get_mine_studio_dir()
    engine_path = os.path.join(mine_studio_dir, "engine", "build", "libs", "mcprec-6.13.jar")
    
    if os.path.exists(engine_path):
        _engine_checked = True
        _engine_exists = True
        return
    
    print(f"Engine not found at {engine_path}")
    print("Auto-downloading simulator engine...")
    try:
        os.makedirs(os.path.dirname(engine_path), exist_ok=True)
        minestudio_entry.download_engine()
        if not os.path.exists(engine_path):
            print(f"ERROR: Engine download failed", file=sys.stderr)
            sys.exit(1)
        print(f"✓ Engine downloaded successfully")
        _engine_checked = True
        _engine_exists = True
    except Exception as e:
        print(f"ERROR: Failed to download engine: {e}", file=sys.stderr)
        sys.exit(1)

minestudio_entry.check_engine = patched_check_engine

# Now import after patching
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, RecordCallback, JudgeResetCallback
from minestudio.models.steve_one import SteveOnePolicy
from MCU_benchmark.utility.read_conf import convert_yaml_to_callbacks

def extract_task_info(task_config_path):
    """Extract commands and task description from YAML config using the utility function"""
    try:
        # Use the existing utility function
        commands, task_dict = convert_yaml_to_callbacks(task_config_path)
        text = task_dict.get('text', '')
        task_name = task_dict.get('name', Path(task_config_path).stem)
        
        # Fallback to filename-based description if text is empty
        if not text:
            text = task_name.replace('_', ' ')
        
        return commands, text, task_name
    except Exception as e:
        # Fallback parsing if utility function fails
        print(f"Warning: Could not parse YAML with utility function: {e}")
        with open(task_config_path, 'r', encoding='utf-8') as f:
            yaml_content = f.read()
        
        lines = yaml_content.splitlines()
        commands = []
        text = ''
        
        for line in lines:
            if line.startswith('-'):
                command = line.strip('- ').strip()
                commands.append(command)
            elif line.startswith('text:'):
                text = line.strip('text: ').strip()
        
        task_name = Path(task_config_path).stem
        if not text:
            text = task_name.replace('_', ' ')
        
        return commands, text, task_name

def run_task_evaluation(task_config_path, output_dir, model, device, max_steps=600):
    """Run evaluation for a single task"""
    
    # Extract task info using utility function
    commands, task_text, task_name = extract_task_info(task_config_path)
    
    print(f"\n{'='*60}")
    print(f"Task: {task_name}")
    print(f"Description: {task_text}")
    print(f"Commands: {commands}")
    print(f"{'='*60}\n")
    
    # Create output directory for this task
    task_output_dir = Path(output_dir) / task_name
    task_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check if already completed
    episode_1 = task_output_dir / "episode_1.mp4"
    if episode_1.exists():
        print(f"✓ Task {task_name} already completed (video exists)")
        return True
    
    # Setup environment
    env = None
    try:
        java_mem = os.environ.get("MINECRAFT_MAX_MEM", "512M")
        env = MinecraftSim(
            obs_size=(128, 128),
            max_mem=java_mem,
            callbacks=[
                CommandsCallback(commands=commands),
                JudgeResetCallback(time_limit=max_steps),
                RecordCallback(
                    record_path=str(task_output_dir),
                    fps=30,
                    frame_type="pov"
                ),
            ]
        )
        
        # Prepare model condition
        print("Preparing model condition...")
        condition = model.prepare_condition({
            'cond_scale': 4.0,
            'text': task_text
        })
        state_in = model.initial_state(condition, 1)
        
        # Reset environment
        print("Resetting environment...")
        obs, info = env.reset()
        
        # Run episode
        print(f"Running episode (max {max_steps} steps)...")
        for step in range(max_steps):
            try:
                # Get action from STEVE-1
                action, state_in = model.get_steve_action(
                    condition,
                    obs,
                    state_in,
                    input_shape='*',
                    deterministic=False
                )
                
                # Convert to numpy
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
                if (step + 1) % 100 == 0:
                    print(f"  Step {step + 1}/{max_steps}")
                
                # Check for early termination
                if terminated or truncated:
                    print(f"  ✓ Episode completed at step {step + 1}")
                    break
                    
            except KeyboardInterrupt:
                print("\n  Interrupted by user")
                return False
            except Exception as e:
                print(f"  ✗ Error at step {step}: {e}")
                return False
        
        print(f"✓ Task {task_name} completed")
        print(f"  Video saved to: {task_output_dir}/episode_1.mp4")
        return True
        
    except Exception as e:
        print(f"✗ Error during task {task_name}: {e}")
        import traceback
        traceback.print_exc()
        return False
        
    finally:
        if env is not None:
            env.close()
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

def main():
    parser = argparse.ArgumentParser(
        description='Run AgentBeats/MCU evaluation with STEVE-1'
    )
    parser.add_argument(
        '--task',
        type=str,
        default=None,
        help='Specific task name (e.g., "collect_wood") or None for all tasks'
    )
    parser.add_argument(
        '--difficulty',
        type=str,
        choices=['simple', 'hard'],
        default='simple',
        help='Task difficulty level'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=600,
        help='Maximum steps per episode'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default=None,
        help='Output directory (default: MCU_benchmark/output/steve_{difficulty})'
    )
    parser.add_argument(
        '--device',
        type=str,
        choices=['auto', 'cuda', 'mps', 'cpu'],
        default='auto',
        help='Device to use (auto detects best available)'
    )
    
    args = parser.parse_args()
    
    # Detect device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device
    
    print("=" * 60)
    print("AgentBeats/MCU Evaluation with STEVE-1")
    print("=" * 60)
    print(f"Device: {device}")
    print(f"Difficulty: {args.difficulty}")
    print(f"Max steps: {args.max_steps}")
    
    # Setup paths
    base_dir = Path(__file__).parent
    task_configs_dir = base_dir / "MCU_benchmark" / "task_configs" / args.difficulty
    
    if not task_configs_dir.exists():
        print(f"✗ Task configs directory not found: {task_configs_dir}")
        sys.exit(1)
    
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = base_dir / "MCU_benchmark" / "output" / f"steve_{args.difficulty}"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output directory: {output_dir}")
    
    # Load STEVE-1 model
    print("\nLoading STEVE-1 model...")
    print("This will download ~2GB on first run. Please wait...")
    try:
        model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to(device)
        model.eval()
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"✗ Error loading model: {e}")
        sys.exit(1)
    
    # Find tasks to run
    if args.task:
        # Single task
        task_file = task_configs_dir / f"{args.task}.yaml"
        if not task_file.exists():
            print(f"✗ Task file not found: {task_file}")
            sys.exit(1)
        task_files = [task_file]
    else:
        # All tasks
        task_files = sorted(task_configs_dir.glob("*.yaml"))
        print(f"\nFound {len(task_files)} tasks to evaluate")
    
    # Run evaluations
    print(f"\nStarting evaluation...")
    print("-" * 60)
    
    completed = 0
    failed = 0
    
    try:
        for task_file in task_files:
            success = run_task_evaluation(
                task_file,
                output_dir,
                model,
                device,
                args.max_steps
            )
            if success:
                completed += 1
            else:
                failed += 1
            
            # Small delay between tasks
            import time
            time.sleep(2)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
    
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
    
    finally:
        # Cleanup
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Summary
    print("\n" + "=" * 60)
    print("Evaluation Summary")
    print("=" * 60)
    print(f"Completed: {completed}")
    print(f"Failed: {failed}")
    print(f"Total: {len(task_files)}")
    print(f"Output directory: {output_dir}")
    print("=" * 60)

if __name__ == "__main__":
    main()

