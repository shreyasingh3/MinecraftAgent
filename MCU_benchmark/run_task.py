import argparse
import torch
import os
import sys
import traceback
import gc
import numpy as np
import signal
import atexit

# Monkey-patch check_engine BEFORE importing MinecraftSim to fix non-interactive Docker issue
# This ensures the installed package version also gets the fix
from minestudio.simulator import entry as minestudio_entry
from minestudio.utils import get_mine_studio_dir

# Cache engine check result to avoid repeated file system checks
_engine_checked = False
_engine_exists = None

def patched_check_engine():
    """Patched version that auto-downloads in non-interactive environments"""
    global _engine_checked, _engine_exists
    
    # If we've already checked and engine exists, skip entirely (fast path)
    if _engine_checked and _engine_exists:
        return
    
    mine_studio_dir = get_mine_studio_dir()
    engine_path = os.path.join(mine_studio_dir, "engine", "build", "libs", "mcprec-6.13.jar")
    
    # Always check if engine exists (even in Docker, in case image wasn't built correctly)
    if os.path.exists(engine_path):
        _engine_checked = True
        _engine_exists = True
        return
    
    # Engine is missing - need to download
    print(f"Engine not found at {engine_path}")
    print(f"MineStudio directory: {mine_studio_dir}")
    
    # Check if we're in a non-interactive environment (Docker, CI, etc.)
    is_interactive = sys.stdin.isatty() if hasattr(sys.stdin, 'isatty') else False
    auto_download = os.environ.get('AUTO_DOWNLOAD_ENGINE', '').lower() in ('1', 'true', 'yes')
    
    if not is_interactive or auto_download:
        print("Detecting missing simulator engine. Auto-downloading from huggingface (non-interactive mode)...")
        print(f"Downloading to: {mine_studio_dir}")
        try:
            # Ensure directory exists
            os.makedirs(os.path.dirname(engine_path), exist_ok=True)
            
            minestudio_entry.download_engine()
            
            # Verify it was downloaded
            if not os.path.exists(engine_path):
                print(f"ERROR: Engine download completed but file not found at {engine_path}", file=sys.stderr)
                print(f"Checking if directory exists: {os.path.exists(os.path.dirname(engine_path))}", file=sys.stderr)
                if os.path.exists(os.path.dirname(engine_path)):
                    print(f"Files in directory: {os.listdir(os.path.dirname(engine_path))}", file=sys.stderr)
                sys.exit(1)
            
            print(f"Engine successfully downloaded to {engine_path}")
            _engine_checked = True
            _engine_exists = True
        except Exception as e:
            print(f"ERROR: Failed to download engine: {e}", file=sys.stderr)
            import traceback
            traceback.print_exc()
            sys.exit(1)
    else:
        try:
            response = input("Detecting missing simulator engine, do you want to download it from huggingface (Y/N)?\n")
            if response == 'Y' or response == 'y':
                minestudio_entry.download_engine()
                _engine_checked = True
                _engine_exists = True
            else:
                sys.exit(0)
        except (EOFError, KeyboardInterrupt):
            # If input fails (non-interactive), auto-download
            print("Detecting missing simulator engine. Auto-downloading from huggingface (non-interactive mode)...")
            minestudio_entry.download_engine()
            _engine_checked = True
            _engine_exists = True

# Apply the monkey patch
minestudio_entry.check_engine = patched_check_engine

# Now import MinecraftSim (it will use our patched check_engine)
from minestudio.simulator import MinecraftSim
from minestudio.simulator.callbacks import CommandsCallback, RecordCallback, SpeedTestCallback, SummonMobsCallback, MaskActionsCallback, RewardsCallback, JudgeResetCallback, FastResetCallback
from minestudio.models import VPTPolicy, load_vpt_policy
# from minestudio.online.run.commands import CommandsCallback
from minestudio.models.steve_one import SteveOnePolicy, load_steve_one_policy

def extract_info(yaml_content, filename):
    lines = yaml_content.splitlines()
    commands = []
    text = ''

    for line in lines:
        if line.startswith('-'):
            command = line.strip('- ').strip()
            commands.append(command)
        elif line.startswith('text:'):
            text = line.strip('text: ').strip()

    filename = filename[:-5].replace('_', ' ')

    print("File:", filename)
    print("Commands:", commands)
    print("Text:", text)
    print("-" * 50)
    return commands, filename

def get_video(commands, text, record_path, model=None, device=None):
    """
    Generate video for a task. If model is provided, reuse it to save memory.
    """
    # Select an appropriate device: prefer CUDA if available, otherwise fall
    # back to Apple Metal (MPS) on macOS, and finally to CPU.
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    # Load model if not provided (for first task)
    model_provided = model is not None
    if model is None:
        print(f"Loading STEVE-1 policy on device: {device}")
        model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to(device)
        model.eval()
    else:
        print(f"Reusing STEVE-1 policy on device: {device}")

    # Limit Java memory to 2G to prevent OOM (can be overridden via MINECRAFT_MAX_MEM env var)
    env = None
    try:
        # Use environment variable for Java memory, default to 512M for maximum headroom
        java_mem = os.environ.get("MINECRAFT_MAX_MEM", "512M")
        env = MinecraftSim(
            obs_size=(128, 128), 
            max_mem=java_mem,
            callbacks=[
                CommandsCallback(commands),
                JudgeResetCallback(600),
                RecordCallback(record_path=record_path, fps=30, frame_type="pov"),
            ]
        )

        # Reduce to 1 episode initially to reduce memory pressure
        n = 1
        try:
            # Reset environment FIRST, then prepare model inputs to spread memory allocation
            print("Resetting environment...")
            try:
                # Cleanup before reset
                gc.collect()
                import time
                time.sleep(1)
                
                obs, info = env.reset()
                print("Environment reset complete, waiting for memory to settle...")
            except Exception as e:
                print(f"Error during reset: {e}", file=sys.stderr)
                gc.collect()
                import time
                time.sleep(2)
                raise
            
            # CRITICAL: Wait longer and cleanup aggressively after reset
            import time
            print("Waiting 8 seconds for memory to settle after reset...")
            time.sleep(8)  # Increased delay to let Java process stabilize
            gc.collect()
            print("Additional 3 second wait...")
            time.sleep(3)  # Additional delay
            gc.collect()  # One more cleanup
            
            print("Preparing model condition and initial state...")
            # NOW prepare condition and state_in AFTER reset settles
            condition = model.prepare_condition(
                {
                    'cond_scale': 4.0,
                    'text': text
                }
            )
            # Cleanup after condition preparation
            gc.collect()
            time.sleep(1)
            
            state_in = model.initial_state(condition, 1)
            # Cleanup after state initialization
            gc.collect()
            time.sleep(1)
            
            print("Memory settled, starting episodes...")
            
            for episode in range(n): 
                print(f"Starting episode {episode + 1}/{n}")
                for i in range(600):
                    # Use get_steve_action which is the proper API for SteveOnePolicy
                    # It handles batching and unbatching automatically
                    try:
                        # Extra aggressive cleanup before first inference to prevent OOM
                        if i == 0:
                            print("Performing aggressive memory cleanup before first step...")
                            gc.collect()
                            import time
                            time.sleep(1)  # Give system time to free memory
                            if torch.cuda.is_available():
                                torch.cuda.empty_cache()
                        
                        action, state_in = model.get_steve_action(condition, obs, state_in, input_shape='*', deterministic=False)
                    except (KeyError, TypeError, AttributeError) as e:
                        # Fallback: some package versions might have different APIs
                        # The installed package version expects condition inside input dict (based on error)
                        input_dict = {'image': obs['image'], 'condition': condition}
                        if isinstance(input_dict['image'], np.ndarray):
                            input_dict['image'] = torch.from_numpy(input_dict['image']).unsqueeze(0).unsqueeze(0).to(model.device)
                        elif isinstance(input_dict['image'], torch.Tensor):
                            input_dict['image'] = input_dict['image'].unsqueeze(0).unsqueeze(0).to(model.device)
                        
                        # Try different calling patterns based on package version
                        try:
                            # Installed package version: condition inside input dict
                            latents, state_in = model(input=input_dict, state_in=state_in)
                        except (KeyError, TypeError):
                            try:
                                # Standard call: condition as separate parameter
                                input_dict_no_cond = {'image': input_dict['image']}
                                latents, state_in = model(condition, input=input_dict_no_cond, state_in=state_in)
                            except (KeyError, TypeError):
                                # Last resort: all keyword arguments
                                input_dict_no_cond = {'image': input_dict['image']}
                                latents, state_in = model(condition=condition, input=input_dict_no_cond, state_in=state_in)
                        
                        # Sample action
                        action = model.pi_head.sample(latents['pi_logits'], deterministic=False)
                        # Unbatch: remove batch and time dimensions
                        action = {k: v[0][0] if len(v.shape) > 2 else v[0] for k, v in action.items()}
                    
                    # Convert to numpy if needed and move off GPU immediately
                    if isinstance(action, dict):
                        # Convert tensors to numpy and ensure they're on CPU
                        action_numpy = {}
                        for k, v in action.items():
                            if isinstance(v, torch.Tensor):
                                action_numpy[k] = v.cpu().numpy()
                            else:
                                action_numpy[k] = v
                        action = action_numpy
                        del action_numpy  # Clean up temporary dict
                    
                    # Force garbage collection periodically to prevent memory buildup
                    if i % 50 == 0 and i > 0:
                        gc.collect()
                    
                    # Critical: Wrap first step in try-catch as it's most likely to OOM
                    try:
                        obs, reward, terminated, truncated, info = env.step(action)
                        if i == 0:
                            print("First step completed successfully!")
                    except MemoryError as e:
                        print(f"Memory error on step {i}: {e}", file=sys.stderr)
                        gc.collect()
                        if torch.cuda.is_available():
                            torch.cuda.empty_cache()
                        raise
                    
                    # Check for early termination
                    if terminated or truncated:
                        print(f"Episode {episode + 1} terminated early at step {i}")
                        break
                    
                    # Progress logging every 100 steps
                    if (i + 1) % 100 == 0:
                        print(f"Episode {episode + 1}, Step {i + 1}/600")

                if episode < n - 1:
                    obs, info = env.reset()
            print(f"Completed {n} episodes successfully")
        except Exception as e:
            print(f"Error during execution: {e}", file=sys.stderr)
            traceback.print_exc()
            raise
    finally:
        # Always try to close the environment, even if initialization failed
        if env is not None:
            try:
                env.close()
            except Exception as e:
                print(f"Warning: Error closing environment: {e}", file=sys.stderr)
        
        # Aggressive memory cleanup
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        # Clean up model only if we created it (not if it was provided)
        if not model_provided and model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

# Global cleanup handler
_cleanup_resources = []

def register_cleanup(resource):
    """Register a resource for cleanup on exit"""
    _cleanup_resources.append(resource)

def cleanup_all():
    """Clean up all registered resources"""
    for resource in _cleanup_resources:
        try:
            if hasattr(resource, 'close'):
                resource.close()
            elif hasattr(resource, '__del__'):
                del resource
        except:
            pass
    _cleanup_resources.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Register cleanup handlers
atexit.register(cleanup_all)

def signal_handler(signum, frame):
    """Handle signals gracefully"""
    print(f"\nReceived signal {signum}, cleaning up...", file=sys.stderr)
    cleanup_all()
    sys.exit(1)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Minecraft tasks')
    parser.add_argument('--difficulty', type=str, help='Difficulty level (simple or hard)')
    parser.add_argument('--max-tasks', type=int, default=1, help='Maximum number of tasks to process (default: 1 for memory testing, use -1 for all)')
    args = parser.parse_args()

    difficulty = args.difficulty
    if difficulty not in ['simple', 'hard']:
        print("Invalid difficulty level. Please choose 'simple' or 'hard'.")
        exit()

    # Print memory configuration for debugging
    java_mem = os.environ.get("MINECRAFT_MAX_MEM", "512M")
    print(f"Java Minecraft memory limit: {java_mem}")
    print(f"Python device: {'cuda' if torch.cuda.is_available() else 'cpu'}")
    
    # Try to print current memory usage if psutil is available
    try:
        import psutil
        process = psutil.Process()
        mem_info = process.memory_info()
        print(f"Current Python process memory: {mem_info.rss / 1024 / 1024 / 1024:.2f} GB")
    except ImportError:
        pass
    
    directory = f'./task_configs/{difficulty}'

    # Use a local, writable directory for saving recorded episodes instead of a
    # hard-coded cluster path like `/nfs-shared-2/...`, which may be read-only
    # or not exist on the current machine.
    base_output_dir = os.path.join(os.path.dirname(__file__), "output", f"steve_{difficulty}")
    os.makedirs(base_output_dir, exist_ok=True)

    # Select device once
    if torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"

    # Load model once and reuse across all tasks to save memory
    print(f"Loading STEVE-1 policy on device: {device} (will be reused for all tasks)")
    model = None
    try:
        # Aggressive cleanup before model loading
        gc.collect()
        import time
        time.sleep(2)
        
        model = SteveOnePolicy.from_pretrained("CraftJarvis/MineStudio_STEVE-1.official").to(device)
        model.eval()
        
        # Critical: Aggressive cleanup and delay after model loading
        print("Model loaded, performing aggressive memory cleanup...")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        time.sleep(5)  # Give system time to free any temporary allocations
        print("Memory cleanup complete, ready to process tasks")

        # Process tasks - limit to first incomplete task for initial memory testing
        tasks_processed = 0
        max_tasks_for_test = float('inf') if args.max_tasks == -1 else args.max_tasks  # -1 means process all
        
        for filename in sorted(os.listdir(directory)):  # Sort for consistent ordering
            if filename.endswith('.yaml'):
                file_path = os.path.join(directory, filename)
                with open(file_path, 'r', encoding='utf-8') as file:
                    yaml_content = file.read()
                
                # `display_text` is a human-readable description derived from the
                # filename, used to condition STEVE-1; `video_dir_name` is a safe
                # directory name for saving recordings.
                commands, display_text = extract_info(yaml_content, filename)
                video_dir_name = filename[:-5]  # e.g. "craft_the_crafting_table"
                record_dir = os.path.join(base_output_dir, video_dir_name)

                # Check if task is already completed by looking for video files
                # For now, only check episode_1 since we're running 1 episode
                episode_1 = os.path.join(record_dir, "episode_1.mp4")
                if os.path.exists(episode_1):
                    print(f"Task {video_dir_name} already completed (video exists)")
                    continue
                
                # Process only first incomplete task initially to test memory stability
                if tasks_processed >= max_tasks_for_test:
                    print(f"Processed {tasks_processed} task(s). Stopping here (--max-tasks limit reached).")
                    print("To process all tasks, use --max-tasks -1")
                    break
                
                # Use the human-readable description to condition the policy.
                print("input text", video_dir_name)
                try:
                    get_video(commands, display_text, record_dir, model=model, device=device)
                    print(f"Successfully completed task: {video_dir_name}")
                    tasks_processed += 1
                except KeyboardInterrupt:
                    print(f"Interrupted during task: {video_dir_name}")
                    sys.exit(1)
                except MemoryError as e:
                    print(f"Out of memory error during task {video_dir_name}: {e}", file=sys.stderr)
                    print("This may indicate insufficient Docker memory limits. Try increasing --memory in docker-run.sh", file=sys.stderr)
                    traceback.print_exc()
                    # Aggressive cleanup after OOM
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    import time
                    time.sleep(2)  # Longer delay after OOM
                    # Continue with next task instead of crashing
                    continue
                except Exception as e:
                    print(f"Error processing task {video_dir_name}: {e}", file=sys.stderr)
                    traceback.print_exc()
                    # Continue with next task instead of crashing
                    continue
                finally:
                    # Always cleanup between tasks to prevent memory accumulation
                    gc.collect()
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    # Small delay to allow cleanup
                    import time
                    time.sleep(1)
    finally:
        # Clean up model at the end
        if model is not None:
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()