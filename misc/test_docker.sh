#!/bin/bash
# Quick test to see if Docker is working

echo "Testing Docker connection..."
echo ""

# Test 1: Docker command exists
if ! command -v docker &> /dev/null; then
    echo "✗ Docker command not found"
    exit 1
fi
echo "✓ Docker command found"

# Test 2: Docker daemon responding
echo -n "Testing Docker daemon... "
if docker ps &> /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
    echo ""
    echo "Docker daemon is not responding!"
    echo "Make sure Docker Desktop is running and fully started."
    exit 1
fi

# Test 3: Check if image exists
echo -n "Checking for image 'mcu-benchmark-cpu'... "
if docker image inspect mcu-benchmark-cpu &> /dev/null 2>&1; then
    echo "✓ Found"
else
    echo "✗ Not found (will be built on first run)"
fi

# Test 4: Try to run a simple container
echo -n "Testing container execution... "
if docker run --rm --platform linux/amd64 mcu-benchmark-cpu echo "test" &> /dev/null 2>&1; then
    echo "✓ OK"
else
    echo "✗ FAILED"
    echo ""
    echo "Cannot run containers. Possible issues:"
    echo "  - Image doesn't exist (run ./run.sh to build it)"
    echo "  - Docker Desktop not fully started"
    echo "  - Platform emulation issues"
    exit 1
fi

echo ""
echo "============================================================"
echo "✓ All Docker tests passed!"
echo "============================================================"
echo ""
echo "You can now run: ./run.sh"

