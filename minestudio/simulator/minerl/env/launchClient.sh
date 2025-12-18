#!/bin/bash

replaceable=0
port=0
seed="NONE"
maxMem="2G"
device="egl"
fatjar=build/libs/mcprec-6.13.jar

while [ $# -gt 0 ]
do
    case "$1" in
        -replaceable) replaceable=1;;
        -port) port="$2"; shift;;
        -seed) seed="$2"; shift;;
        -maxMem) maxMem="$2"; shift;;
        -device) device="$2"; shift;;
        -fatjar) fatjar="$2"; shift;;
        *) echo >&2 \
            "usage: $0 [-replaceable] [-port <port>] [-seed <seed>] [-maxMem <maxMem>] [-device <device>] [-fatjar <fatjar>]"
            exit 1;;
    esac
    shift
done
  
if ! [[ $port =~ ^-?[0-9]+$ ]]; then
    echo "Port value should be numeric"
    exit 1
fi


if [ \( $port -lt 0 \) -o \( $port -gt 65535 \) ]; then
    echo "Port value out of range 0-65535"
    exit 1
fi

# Detect OS - macOS doesn't need xvfb-run or vglrun
OS="$(uname -s)"
case "$OS" in
    Darwin*)
        # macOS - run Java directly (has native display support)
        # LWJGL 3.x automatically extracts native libraries from the JAR
        # If the JAR doesn't contain macOS libraries, this will fail
        # In that case, you may need to use Rosetta 2 or Docker
        
        # Detect architecture
        ARCH=$(uname -m)
        
        # Check if we should use Rosetta 2 (for x86_64 JARs on ARM64 Mac)
        USE_ROSETTA="${USE_ROSETTA:-0}"
        
        if [ "$ARCH" = "arm64" ] && [ "$USE_ROSETTA" = "1" ]; then
            # Use Rosetta 2 to run x86_64 Java
            # This requires x86_64 Java to be installed
            # Install with: arch -x86_64 /usr/local/bin/brew install --cask temurin8
            echo "Using Rosetta 2 to run x86_64 Java..." >&2
            arch -x86_64 java -Xmx$maxMem \
                 -Dfile.encoding=UTF-8 \
                 -Duser.country=US \
                 -Duser.language=en \
                 -jar "$fatjar" --envPort=$port
        else
            # Try native execution - let LWJGL handle library extraction
            # LWJGL will extract to a temp directory automatically
            java -Xmx$maxMem \
                 -Dfile.encoding=UTF-8 \
                 -Duser.country=US \
                 -Duser.language=en \
                 -jar "$fatjar" --envPort=$port
        fi
        ;;
    Linux*)
        # Linux - use xvfb-run for CPU or vglrun for GPU
        if [ "$device" == "cpu" ]; then
            # Check if xvfb-run is available
            if command -v xvfb-run >/dev/null 2>&1; then
                xvfb-run -a java -Xmx$maxMem -jar $fatjar --envPort=$port
            else
                # Fallback: try with DISPLAY variable or run directly
                echo "Warning: xvfb-run not found, attempting direct execution" >&2
                java -Xmx$maxMem -jar $fatjar --envPort=$port
            fi
        else
            # Check if vglrun is available
            if command -v vglrun >/dev/null 2>&1; then
                vglrun -d $device java -Xmx$maxMem -jar $fatjar --envPort=$port
            else
                echo "Warning: vglrun not found, attempting direct execution" >&2
                java -Xmx$maxMem -jar $fatjar --envPort=$port
            fi
        fi
        ;;
    *)
        # Unknown OS - try direct execution
        echo "Warning: Unknown OS $OS, attempting direct execution" >&2
        java -Xmx$maxMem -jar $fatjar --envPort=$port
        ;;
esac

[ $replaceable -gt 0 ]

