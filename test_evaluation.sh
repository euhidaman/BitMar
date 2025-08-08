#!/bin/bash

# Quick Evaluation Test Script
# Tests evaluation on the latest available checkpoint

set -e

echo "🧪 BitMar Evaluation Quick Test"
echo "================================"

# Find the most recent checkpoint
CHECKPOINT_DIR="checkpoints_100M_dataset"
LATEST_CHECKPOINT=""

if [[ -d "$CHECKPOINT_DIR" ]]; then
    # Look for step-based checkpoints first
    LATEST_STEP=$(find "$CHECKPOINT_DIR" -name "checkpoint_step_*.pt" -printf '%f\n' 2>/dev/null | \
                  sed 's/checkpoint_step_\([0-9]*\)_tokens_[0-9]*.pt/\1/' | \
                  sort -n | tail -1)

    if [[ -n "$LATEST_STEP" ]]; then
        LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "checkpoint_step_${LATEST_STEP}_*.pt" | head -1)
        echo "📁 Found step checkpoint: $(basename "$LATEST_CHECKPOINT")"
    else
        # Look for epoch checkpoints
        LATEST_EPOCH=$(find "$CHECKPOINT_DIR" -name "checkpoint_epoch_*.pt" -printf '%f\n' 2>/dev/null | \
                       sed 's/checkpoint_epoch_\([0-9]*\)_tokens_[0-9]*.pt/\1/' | \
                       sort -n | tail -1)

        if [[ -n "$LATEST_EPOCH" ]]; then
            LATEST_CHECKPOINT=$(find "$CHECKPOINT_DIR" -name "checkpoint_epoch_${LATEST_EPOCH}_*.pt" | head -1)
            echo "📁 Found epoch checkpoint: $(basename "$LATEST_CHECKPOINT")"
        else
            # Try latest_checkpoint.pt
            if [[ -f "$CHECKPOINT_DIR/latest_checkpoint.pt" ]]; then
                LATEST_CHECKPOINT="$CHECKPOINT_DIR/latest_checkpoint.pt"
                echo "📁 Found latest checkpoint: $(basename "$LATEST_CHECKPOINT")"
            fi
        fi
    fi
fi

if [[ -z "$LATEST_CHECKPOINT" ]]; then
    echo "❌ No checkpoint found in $CHECKPOINT_DIR"
    echo "💡 Start training first with:"
    echo "   python3 train_100M_tokens.py --config configs/bitmar_100M_tokens.yaml --device cuda:0 --save_every_n_steps 500"
    exit 1
fi

echo "✅ Testing evaluation on: $LATEST_CHECKPOINT"
echo ""

# Check file size
if [[ -f "$LATEST_CHECKPOINT" ]]; then
    SIZE_MB=$(stat -f%z "$LATEST_CHECKPOINT" 2>/dev/null || stat -c%s "$LATEST_CHECKPOINT" 2>/dev/null || echo "0")
    SIZE_MB=$((SIZE_MB / 1024 / 1024))
    echo "📊 Checkpoint size: ${SIZE_MB} MB"
else
    echo "❌ Checkpoint file not found: $LATEST_CHECKPOINT"
    exit 1
fi

# Test evaluation
echo "🚀 Starting evaluation test..."
echo ""

# Create test output directory
TEST_OUTPUT="evaluation_test_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$TEST_OUTPUT"

echo "📂 Test output directory: $TEST_OUTPUT"
echo ""

# Run fast evaluation test
echo "⚡ Running fast evaluation test..."
./run_evaluation.sh --model_path "$LATEST_CHECKPOINT" --eval_type fast --output_dir "$TEST_OUTPUT"

# Check results
echo ""
echo "📊 Evaluation Test Results:"
echo "=========================="

if [[ -d "$TEST_OUTPUT" ]]; then
    echo "✅ Test output directory created"

    # Count result files
    RESULT_FILES=$(find "$TEST_OUTPUT" -name "*.json" | wc -l)
    echo "📄 Result files generated: $RESULT_FILES"

    if [[ $RESULT_FILES -gt 0 ]]; then
        echo "✅ Evaluation test PASSED!"
        echo ""
        echo "📁 Result files:"
        find "$TEST_OUTPUT" -name "*.json" | head -5 | while read file; do
            echo "  • $(basename "$file")"
        done

        if [[ $RESULT_FILES -gt 5 ]]; then
            echo "  • ... and $((RESULT_FILES - 5)) more"
        fi
    else
        echo "❌ No result files generated"
        echo "🔍 Check logs for errors"
    fi
else
    echo "❌ Test output directory not created"
fi

echo ""
echo "🎉 Evaluation test completed!"
echo "💡 To run full evaluation: ./run_evaluation.sh --model_path \"$LATEST_CHECKPOINT\" --eval_type full"
