#!/bin/bash
#####################################################################
# Run All Human Activities Benchmarks - Qwen3-VL (2B, 4B, 8B, 32B)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "Running All Human Activities Benchmarks - Qwen3-VL"
echo "======================================================================"

# Define all scripts to run (Think + NoThink versions)
SCRIPTS=(
    # 2B Models
    "visual_2b.sh"
    "visual_nothink_2b.sh"
    "text_2b.sh"
    "text_nothink_2b.sh"
    # 4B Models
    "visual_4b.sh"
    "visual_nothink_4b.sh"
    "text_4b.sh"
    "text_nothink_4b.sh"
    # 8B Models
    "visual_8b.sh"
    "visual_nothink_8b.sh"
    "text_8b.sh"
    "text_nothink_8b.sh"
    # 32B Models
    "visual_32b.sh"
    "visual_nothink_32b.sh"
    "text_32b.sh"
    "text_nothink_32b.sh"
)

# Run each script
for script in "${SCRIPTS[@]}"; do
    SCRIPT_PATH="${SCRIPT_DIR}/${script}"

    if [ -f "$SCRIPT_PATH" ]; then
        echo ""
        echo "======================================================================"
        echo "Starting: ${script}"
        echo "======================================================================"
        bash "$SCRIPT_PATH"
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
            echo "Warning: ${script} failed with exit code $EXIT_CODE"
        fi
    else
        echo "Warning: Script not found: $SCRIPT_PATH"
    fi
done

echo ""
echo "======================================================================"
echo "All Qwen3-VL Human Activities benchmarks completed!"
echo "======================================================================"
