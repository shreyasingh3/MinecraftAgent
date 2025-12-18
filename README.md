# Minecraft Agents - STEVE-1 Evaluation System

A complete system for evaluating Minecraft AI agents using the MCU (Minecraft Understanding) benchmark. This repository integrates STEVE-1 (a vision-language model agent) with the AgentBeats evaluation framework to run and score agent performance on Minecraft tasks.

## Table of Contents

- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Agents Explained](#agents-explained)
- [Requirements](#requirements)
- [Docker Setup](#docker-setup)
- [Quick Start](#quick-start)
- [Documentation](#documentation)
- [Project Structure](#project-structure)
- [Usage Examples](#usage-examples)
- [Troubleshooting](#troubleshooting)

## Overview

This project provides:

1. **Agent Evaluation Framework** - The "Green Agent" (MCU Benchmark system) that manages task execution, video recording, and scoring
2. **AI Agent** - The "White Agent" (STEVE-1) that performs tasks in Minecraft using vision-language understanding
3. **Automated Scoring** - Vision-language model (VLM) based evaluation of agent performance
4. **Docker Environment** - Complete containerized setup for consistent execution across platforms

## System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Docker Container                         │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Green Agent (MCU Benchmark)                         │  │
│  │  - Task Management                                    │  │
│  │  - Environment Setup                                  │  │
│  │  - Video Recording                                    │  │
│  │  - Performance Evaluation                            │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  White Agent (STEVE-1)                               │  │
│  │  - Vision-Language Model                             │  │
│  │  - Task Understanding                                │  │
│  │  - Action Execution                                  │  │
│  └───────────────────────────────────────────────────────┘  │
│                          │                                   │
│                          ▼                                   │
│  ┌───────────────────────────────────────────────────────┐  │
│  │  Minecraft Simulator (MineStudio)                    │  │
│  │  - Environment Rendering                              │  │
│  │  - Physics Simulation                                │  │
│  │  - State Management                                  │  │
│  └───────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────┘
```

## Agents Explained

### Green Agent (MCU Benchmark / Evaluation System)

**What it does:**
- **Task Management**: Loads task configurations from YAML files, extracts task descriptions and commands
- **Environment Orchestration**: Sets up Minecraft environments with appropriate callbacks (recording, reset conditions, command execution)
- **Video Recording**: Captures agent behavior as MP4 videos for later analysis
- **Performance Evaluation**: Uses vision-language models to score agent performance across multiple dimensions:
  - Task Progress
  - Material Selection and Usage
  - Action Control
  - Error Recognition and Correction
  - Creative Attempts
  - Task Completion Efficiency

**Key Components:**
- `MCU_benchmark/run_task.py` - Main evaluation runner
- `MCU_benchmark/utility/` - Task configuration parsing and callbacks
- `MCU_benchmark/auto_eval/` - Automated video scoring system
- `MCU_benchmark/task_configs/` - Task definitions (simple, hard, compositional)

**Why "Green":** The evaluation framework is the infrastructure/system that manages everything (like a green traffic light - go/system ready).

### White Agent (STEVE-1)

**What it does:**
- **Vision-Language Understanding**: Processes visual observations (first-person Minecraft view) and natural language task descriptions
- **Decision Making**: Generates action sequences to complete tasks based on visual input and text instructions
- **Task Execution**: Performs actions in Minecraft (movement, mining, crafting, building, etc.)
- **Adaptive Behavior**: Can handle various task types from simple collection to complex building tasks

**Key Components:**
- `minestudio/models/steve_one/` - STEVE-1 model implementation
- `run_agentbeats_evaluation.py` - Integration script that connects STEVE-1 to MCU benchmark
- `run_steve1_inference.py` - Standalone inference script for testing

**Model Details:**
- **Architecture**: Vision Transformer (ViT) + Language Model
- **Input**: 128x128 RGB frames + text task description
- **Output**: Action sequences (movement, camera, inventory, crafting)
- **Pre-trained**: Available on HuggingFace (`CraftJarvis/MineStudio_STEVE-1.official`)

**Why "White":** The agent is the "blank slate" that learns and performs tasks (like a white canvas).

### How They Work Together

1. **Green Agent** loads a task configuration (e.g., "collect_wood")
2. **Green Agent** sets up Minecraft environment with recording enabled
3. **Green Agent** passes task description to **White Agent** (STEVE-1)
4. **White Agent** processes visual observations and generates actions
5. **White Agent** executes actions in Minecraft
6. **Green Agent** records the episode as a video
7. **Green Agent** evaluates the video using VLM scoring
8. Results are saved with scores and metrics

## Requirements

### System Requirements

- **Operating System**: macOS, Linux, or Windows (with WSL2)
- **RAM**: 20GB+ available (for Docker)
- **Disk Space**: 10GB+ free space
- **Docker Desktop**: Latest version installed and running

### Software Requirements

- **Docker Desktop** (required)
  - Download: https://www.docker.com/products/docker-desktop/
  - Allocate **20GB RAM** in Docker Desktop settings
  - Enable virtualization in BIOS (if on Windows/Linux)

- **Git** (for cloning repository)

### Python Requirements (Inside Docker)

All Python dependencies are installed automatically in the Docker container:
- Python 3.10
- PyTorch (CPU version)
- MineStudio
- NumPy, OpenCV, Pillow
- HuggingFace Hub
- And more (see `requirements.txt`)

### Java Requirements (Inside Docker)

- OpenJDK 8 (installed automatically in Docker)

## Docker Setup

### Why Docker?

The Minecraft engine (`mcprec-6.13.jar`) was built for Linux and contains Linux native libraries. Docker provides:
- **Cross-platform compatibility** - Works on macOS, Linux, Windows
- **Consistent environment** - Same setup everywhere
- **Isolated execution** - No conflicts with system packages
- **Easy deployment** - One command to run everything

### Docker Architecture

- **Base Image**: Ubuntu 22.04 (Linux x86_64)
- **Platform**: Emulated on macOS ARM64 via Docker Desktop
- **Display**: Xvfb (virtual framebuffer for headless operation)
- **Memory**: 20GB limit (configurable)
- **Volume Mounts**: Project directory mounted for input/output

### Docker Image Contents

The `Dockerfile.cpu` includes:
- Ubuntu 22.04 base
- Python 3.10
- OpenJDK 8
- Xvfb (for headless display)
- PyTorch CPU version
- MineStudio package
- All project code

## Quick Start

### 1. Prerequisites Check

```bash
# Check Docker is installed
docker --version

# Check Docker is running
docker ps
```

If Docker isn't running:
1. Open Docker Desktop
2. Wait for menu bar icon to be solid (not animating)
3. Wait 30 seconds for full initialization

### 2. Run Evaluation

```bash
# Make script executable (if needed)
chmod +x run.sh

# Run evaluation
./run.sh
```

**What happens:**
1. Script waits for Docker to be ready (up to 60 seconds)
2. Builds Docker image if needed (first time: 10-15 minutes)
3. Starts container with Minecraft simulator
4. Runs STEVE-1 on "collect_wood" task
5. Saves video and results

### 3. Check Results

Results are saved to:
```
MCU_benchmark/output/steve_simple/collect_wood/episode_1.mp4
```

## Usage Commands

### Running White Agents to Complete Tasks

White agents (e.g., STEVE-1) are the AI agents that perform tasks in Minecraft. Here are the commands to run them:

#### Using Docker (Recommended)

```bash
# Run a single task using the convenience script
./run.sh

# Or run directly with Docker
docker run --rm --platform linux/amd64 --memory=20g \
  -v $(pwd):/workspace -e AUTO_DOWNLOAD_ENGINE=1 \
  mcu-benchmark-cpu \
  bash -c "cd /workspace && python3 run_agentbeats_evaluation.py --task collect_wood --difficulty simple"
```

#### Using Python Directly (Inside Docker Container)

```bash
# Single task
python3 run_agentbeats_evaluation.py --task collect_wood --difficulty simple

# Multiple tasks (all tasks in a difficulty level)
python3 run_agentbeats_evaluation.py --difficulty simple

# Custom task with specific parameters
python3 run_agentbeats_evaluation.py \
  --task build_house \
  --difficulty hard \
  --max-steps 1200 \
  --device cuda
```

**Available Options:**
- `--task`: Task name (e.g., "collect_wood", "build_house"). Omit to run all tasks.
- `--difficulty`: Task difficulty level (`simple` or `hard`)
- `--max-steps`: Maximum steps per episode (default: 600)
- `--output-dir`: Custom output directory (default: `MCU_benchmark/output/steve_{difficulty}`)
- `--device`: Device to use (`auto`, `cuda`, `mps`, `cpu`)

**Output:** Videos are saved to `MCU_benchmark/output/steve_{difficulty}/{task_name}/episode_1.mp4`

### Running Green Agent to Evaluate White Agents

The Green Agent (MCU Benchmark evaluation system) evaluates white agent performance using vision-language models. Evaluation can be done on individual videos or in batch.

#### Individual Video Evaluation

```bash
cd MCU_benchmark/auto_eval

# Evaluate a single video
python individual_video_rating.py \
  --video_path='../output/steve_simple/collect_wood/episode_1.mp4' \
  --criteria_path='./criteria_files/collect_wood.txt'
```

**Requirements:**
- Set `OPENAI_API_KEY` environment variable
- Video file must exist
- Criteria file must exist in `criteria_files/` directory

**Output:** Results saved to `vlm_rating_res/{task_name}.json`

#### Batch Video Evaluation

```bash
cd MCU_benchmark/auto_eval

# Evaluate all videos in a directory
python batch_video_rating.py \
  --videos_path='../output/steve_simple/' \
  --criteria_files_path='./criteria_files/'
```

**Directory Structure Required:**
```
videos_path/
├── collect_wood/
│   └── episode_1.mp4
├── build_house/
│   └── episode_1.mp4
└── ...

criteria_files_path/
├── collect_wood.txt
├── build_house.txt
└── ...
```

**Output:** 
- Individual results: `vlm_rating_res/{task_name}.json`
- Summary metrics printed to console (averages across all tasks)

#### Video Comparison (Comparing Two Agents)

```bash
cd MCU_benchmark/auto_eval

# Compare two videos side-by-side
python video_comparison.py \
  --video_path_a='./eval_video/build_gate/build_gate_5.mp4' \
  --video_path_b='./eval_video/build_gate/build_gate_7.mp4' \
  --criteria_path='./criteria_files/build_gate.txt'
```

**Output:** Comparison results saved to `vlm_rating_res/{task_name}.json`

### Testing Green Agent's Evaluation Results

To test the green agent's evaluation system on known test cases:

#### Using Provided Test Videos

```bash
cd MCU_benchmark/auto_eval

# Test on individual test videos
python individual_video_rating.py \
  --video_path='./eval_video/build_gate/build_gate_5.mp4' \
  --criteria_path='./criteria_files/build_gate.txt'

python individual_video_rating.py \
  --video_path='./eval_video/build_pillar/build_pillar_0.mp4' \
  --criteria_path='./criteria_files/build_pillar.txt'

python individual_video_rating.py \
  --video_path='./eval_video/combat_spider/combat_spider_7.mp4' \
  --criteria_path='./criteria_files/combat_spider.txt'
```

#### Batch Testing on All Test Videos

```bash
cd MCU_benchmark/auto_eval

# Test on all provided test videos
python batch_video_rating.py \
  --videos_path='./eval_video/' \
  --criteria_files_path='./criteria_files/'
```

**Test Videos Available:**
- `eval_video/build_gate/` - Gate building task videos
- `eval_video/build_pillar/` - Pillar building task videos  
- `eval_video/combat_spider/` - Spider combat task videos

**Expected Output:**
- JSON files with scores (0-10) for each evaluation dimension:
  - Task Progress
  - Action Control
  - Error Recognition and Correction
  - Creative Attempts
  - Task Completion Efficiency
  - Material Selection and Usage

### Reproducing Original Benchmark Results

To reproduce the results from the original MCU benchmark:

#### 1. Run All Tasks for a Difficulty Level

```bash
# Run all simple tasks
python3 run_agentbeats_evaluation.py --difficulty simple

# Run all hard tasks
python3 run_agentbeats_evaluation.py --difficulty hard
```

#### 2. Evaluate All Generated Videos

```bash
cd MCU_benchmark/auto_eval

# Evaluate simple task results
python batch_video_rating.py \
  --videos_path='../output/steve_simple/' \
  --criteria_files_path='./criteria_files/'

# Evaluate hard task results
python batch_video_rating.py \
  --videos_path='../output/steve_hard/' \
  --criteria_files_path='./criteria_files/'
```

#### 3. Compare with Baseline Results

Baseline performance metrics are documented in `docs/baseline.md`. Expected STEVE-1 performance:

**Simple Mode:**
- Task Progress: 31.4%
- Action Control: 31.9%
- Error Recognition: 13.1%
- Creative Attempts: 6.4%
- Task Efficiency: 23.2%
- Material Usage: 35.6%

**Hard Mode:**
- Task Progress: 23.1%
- Action Control: 22.6%
- Error Recognition: 6.9%
- Creative Attempts: 6.0%
- Task Efficiency: 16.6%
- Material Usage: 24.5%

#### 4. Full Benchmark Reproduction Pipeline

```bash
# Step 1: Run all tasks (inside Docker)
docker run --rm --platform linux/amd64 --memory=20g \
  -v $(pwd):/workspace -e AUTO_DOWNLOAD_ENGINE=1 \
  mcu-benchmark-cpu \
  bash -c "cd /workspace && python3 run_agentbeats_evaluation.py --difficulty simple"

# Step 2: Evaluate all videos (requires OPENAI_API_KEY)
cd MCU_benchmark/auto_eval
export OPENAI_API_KEY="your-api-key-here"
python batch_video_rating.py \
  --videos_path='../output/steve_simple/' \
  --criteria_files_path='./criteria_files/'

# Step 3: Review results
ls -la vlm_rating_res/
cat vlm_rating_res/collect_wood.json
```

**Note:** Full benchmark reproduction requires:
- Running all 83 simple tasks or 82 hard tasks
- OpenAI API access for VLM-based evaluation
- Significant compute time (several hours for all tasks)

## Documentation

### Main Documentation Files

- **README.md** (this file) - Overview and quick start
- **MCU_benchmark/auto_eval/README.md** - Video evaluation system documentation
- **docs/** - Additional documentation:
  - `automatic-evaluation.md` - Automated scoring details
  - `baseline.md` - Baseline agent information
  - `quick-benchmark.md` - Quick benchmark guide

### Code Documentation

**Python Scripts:**
- `run_agentbeats_evaluation.py` - Main evaluation script (docstrings explain integration)
- `run_steve1_inference.py` - Simple inference example (docstrings explain usage)
- `MCU_benchmark/run_task.py` - Task execution system (docstrings explain workflow)

**Configuration Files:**
- `MCU_benchmark/task_configs/` - YAML task definitions
  - `simple/` - 83 simple tasks (collection, basic crafting)
  - `hard/` - 82 hard tasks (complex building, combat)
  - `compositional/` - 17 multi-step tasks

### Task Configuration Format

Tasks are defined in YAML files:

```yaml
name: collect_wood
text: Collect wood from trees
commands:
  - /give @p minecraft:iron_axe
  - /time set day
```

- **name**: Task identifier
- **text**: Natural language description for the agent
- **commands**: Minecraft commands to set up the task environment

## Project Structure

```
MinecraftAgents/
├── run.sh                          # Main execution script
├── Dockerfile.cpu                   # Docker image definition
├── requirements.txt                 # Python dependencies
│
├── run_agentbeats_evaluation.py     # STEVE-1 + MCU integration
├── run_steve1_inference.py         # Simple STEVE-1 test
│
├── MCU_benchmark/                   # Green Agent (Evaluation System)
│   ├── run_task.py                 # Main task runner
│   ├── task_configs/               # Task definitions
│   │   ├── simple/                 # 83 simple tasks
│   │   ├── hard/                   # 82 hard tasks
│   │   └── compositional/          # 17 complex tasks
│   ├── auto_eval/                  # Automated scoring
│   │   ├── batch_video_rating.py   # Batch evaluation
│   │   ├── individual_video_rating.py  # Single video scoring
│   │   └── criteria_files/         # Scoring criteria (82 tasks)
│   ├── utility/                    # Helper functions
│   │   ├── read_conf.py            # YAML parsing
│   │   └── task_call.py            # Task execution
│   └── output/                     # Results directory
│       └── steve_simple/           # STEVE-1 results
│
├── minestudio/                     # MineStudio framework
│   ├── models/
│   │   └── steve_one/              # STEVE-1 model code
│   ├── simulator/                  # Minecraft simulator
│   │   ├── entry.py                # MinecraftSim class
│   │   └── callbacks/              # Environment callbacks
│   └── utils/                      # Utilities
│
└── docs/                           # Additional documentation
```

## Usage Examples

### Run Single Task Evaluation

```bash
# Using the main script (collect_wood task)
./run.sh

# Using Python directly (inside Docker)
docker run --rm --platform linux/amd64 --memory=20g \
  -v $(pwd):/workspace -e AUTO_DOWNLOAD_ENGINE=1 \
  mcu-benchmark-cpu \
  bash -c "cd /workspace && python3 run_agentbeats_evaluation.py --task collect_wood --difficulty simple"
```

### Run Multiple Tasks

```bash
# Edit run.sh or create custom script
# Change task name: --task collect_wood
# To: --task build_house
```

### Run Simple Inference Test

```bash
# Test STEVE-1 without full evaluation
docker run --rm --platform linux/amd64 --memory=20g \
  -v $(pwd):/workspace -e AUTO_DOWNLOAD_ENGINE=1 \
  mcu-benchmark-cpu \
  bash -c "cd /workspace && python3 run_steve1_inference.py"
```

### Evaluate Videos

```bash
# After running tasks, evaluate videos
cd MCU_benchmark/auto_eval
python individual_video_rating.py \
  --video_path='../output/steve_simple/collect_wood/episode_1.mp4' \
  --criteria_path='./criteria_files/collect_wood.txt'
```

## Troubleshooting

### Docker Issues

**Problem**: Docker not responding
```bash
# Solution: Restart Docker Desktop
# 1. Quit Docker Desktop completely
# 2. Wait 10 seconds
# 3. Open Docker Desktop
# 4. Wait 30 seconds
# 5. Run: docker ps (should work)
```

**Problem**: Container not starting
```bash
# Check Docker Desktop has enough RAM
# Settings → Resources → Memory → 20GB+

# Check Docker is actually running
docker ps
```

**Problem**: Build fails
```bash
# Clean Docker cache and rebuild
docker system prune -a
./run.sh
```

### Evaluation Issues

**Problem**: "Engine not found"
- The script auto-downloads the Minecraft engine
- Check internet connection
- Check Docker has network access

**Problem**: Out of memory
- Increase Docker Desktop RAM allocation
- Reduce `MINECRAFT_MAX_MEM` in `run.sh` (change `2G` to `1G`)

**Problem**: Task fails
- Check task config exists: `MCU_benchmark/task_configs/simple/collect_wood.yaml`
- Check output directory is writable
- Check Docker volume mount is working

### Performance Tips

- **First run**: Takes 10-15 minutes (Docker build)
- **Subsequent runs**: 5-10 minutes (just evaluation)
- **Video generation**: Adds 1-2 minutes per task
- **Memory usage**: ~15-18GB during evaluation

## Advanced Usage

### Custom Tasks

1. Create YAML file in `MCU_benchmark/task_configs/simple/`
2. Define task name, text, and commands
3. Run evaluation with `--task your_task_name`

### Batch Evaluation

```bash
# Run multiple tasks
for task in collect_wood build_house craft_table; do
  docker run --rm --platform linux/amd64 --memory=20g \
    -v $(pwd):/workspace -e AUTO_DOWNLOAD_ENGINE=1 \
    mcu-benchmark-cpu \
    bash -c "cd /workspace && python3 run_agentbeats_evaluation.py --task $task --difficulty simple"
done
```

### Custom Agent Integration

To use a different agent:
1. Implement agent interface (see `minestudio/models/steve_one/body.py`)
2. Modify `run_agentbeats_evaluation.py` to load your agent
3. Run evaluation as normal

## Citation

Thank you to the creators of MCU, MineStudio, and STEVE-1. We relied heavily on their research. 
https://github.com/CraftJarvis
