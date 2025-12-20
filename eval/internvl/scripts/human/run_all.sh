#!/bin/bash
#####################################################################
# Run All Human Activities Benchmarks - InternVL (4B, 14B, 38B)
#####################################################################

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "======================================================================"
echo "Running All Human Activities Benchmarks - InternVL"
echo "======================================================================"

# Define all scripts to run
SCRIPTS=(
    # 4B
    "visual_4b.sh"
    "visual_nothink_4b.sh"
    "text_4b.sh"
    "text_nothink_4b.sh"
    # 14B
    "visual_14b.sh"
    "visual_nothink_14b.sh"
    "text_14b.sh"
    "text_nothink_14b.sh"
    # 38B
    "visual_38b.sh"
    "visual_nothink_38b.sh"
    "text_38b.sh"
    "text_nothink_38b.sh"
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
echo "All InternVL Human Activities benchmarks completed!"
echo "======================================================================"
