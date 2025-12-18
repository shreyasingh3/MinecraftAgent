#!/bin/bash
# Single script to run STEVE-1 evaluation in Docker
# This is the ONLY script you need

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

IMAGE="mcu-benchmark-cpu"

echo "============================================================"
echo "STEVE-1 Evaluation Runner"
echo "============================================================"
echo ""

# Check Docker command exists
if ! command -v docker &> /dev/null; then
    echo "ERROR: Docker not found!"
    echo "Install Docker Desktop: https://www.docker.com/products/docker-desktop/"
    exit 1
fi

# Wait for Docker to be ready (max 60 seconds)
echo "Waiting for Docker..."
for i in {1..60}; do
    if docker ps &> /dev/null 2>&1; then
        echo "✓ Docker ready"
        break
    fi
    if [ $i -eq 60 ]; then
        echo ""
        echo "ERROR: Docker not responding after 60 seconds"
        echo "Make sure Docker Desktop is running and fully started"
        exit 1
    fi
    sleep 1
done

# Check/build image
echo ""
echo "Checking Docker image..."
if ! docker image inspect "$IMAGE" &> /dev/null 2>&1; then
    echo "Building Docker image (first time, takes 10-15 minutes)..."
    docker build --platform linux/amd64 -f Dockerfile.cpu -t "$IMAGE" .
    echo "✓ Image built"
else
    echo "✓ Image exists"
fi

# Run evaluation
echo ""
echo "Running STEVE-1 evaluation..."
echo "Task: collect_wood (simple)"
echo ""

# Test Docker connection first
echo "Testing Docker connection..."
if ! docker ps &> /dev/null; then
    echo "ERROR: Cannot connect to Docker daemon!"
    echo ""
    echo "Try:"
    echo "  1. Restart Docker Desktop"
    echo "  2. Wait 30 seconds after restart"
    echo "  3. Run: docker ps (should show empty list)"
    echo "  4. Run this script again"
    exit 1
fi
echo "✓ Docker connection OK"
echo ""

# Run with better error handling
set +e  # Don't exit on error, we'll handle it
docker run --rm \
    --platform linux/amd64 \
    --memory=20g \
    --memory-swap=20g \
    --ulimit nofile=65536:65536 \
    --ulimit nproc=4096:4096 \
    -v "$SCRIPT_DIR":/workspace \
    -e DISPLAY=:99 \
    -e JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64 \
    -e MINESTUDIO_DIR=/opt/minestudio \
    -e AUTO_DOWNLOAD_ENGINE=1 \
    -e MINECRAFT_MAX_MEM=2G \
    "$IMAGE" \
    bash -c "
        echo '=== Container started ==='
        echo 'Starting Xvfb...'
        Xvfb :99 -screen 0 1024x768x24 >/dev/null 2>&1 &
        sleep 3
        echo 'Xvfb started'
        echo ''
        echo 'Running STEVE-1 evaluation...'
        cd /workspace
        python3 run_agentbeats_evaluation.py --task collect_wood --difficulty simple
    "

EXIT_CODE=$?
set -e

echo ""
echo "============================================================"
if [ $EXIT_CODE -eq 0 ]; then
    echo "✓ Evaluation complete!"
    echo "============================================================"
    echo "Results: MCU_benchmark/output/steve_simple/collect_wood/"
else
    echo "✗ Evaluation failed (exit code: $EXIT_CODE)"
    echo "============================================================"
    echo ""
    echo "Check the error messages above."
    echo ""
    echo "Common issues:"
    echo "  - Docker Desktop not fully started (wait 30 seconds)"
    echo "  - Not enough memory (need 20GB allocated to Docker)"
    echo "  - Image build incomplete (rebuild with: docker build --platform linux/amd64 -f Dockerfile.cpu -t mcu-benchmark-cpu .)"
    exit $EXIT_CODE
fi
echo ""

