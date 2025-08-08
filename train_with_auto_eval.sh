#!/bin/bash

# Training with Evaluation - Wrapper Script
# This script provides easy options to run training with automatic evaluation

set -e

# Default values
CONFIG="configs/bitmar_100M_tokens.yaml"
DEVICE="cuda:0"
SAVE_EVERY_N_STEPS=""
ENABLE_FAST_EVAL=true
ENABLE_FULL_EVAL=true
REBUILD_CACHE=false

# Function to show usage
show_usage() {
    echo "Usage: $0 [options]"
    echo ""
    echo "Training Options:"
    echo "  --config <path>            Configuration file (default: configs/bitmar_100M_tokens.yaml)"
    echo "  --device <device>          Device to use (default: cuda:0)"
    echo "  --save_every_n_steps <n>   Save checkpoint every N steps (optional)"
    echo "  --rebuild_cache            Rebuild dataset cache"
    echo ""
    echo "Evaluation Options:"
    echo "  --disable_fast_eval        Disable fast evaluation after each epoch"
    echo "  --disable_full_eval        Disable full evaluation at the end"
    echo ""
    echo "Other Options:"
    echo "  --help, -h                 Show this help message"
    echo ""
    echo "Examples:"
    echo "  # Standard training with automatic evaluation"
    echo "  $0"
    echo ""
    echo "  # Training with step-based checkpoints every 1000 steps"
    echo "  $0 --save_every_n_steps 1000"
    echo ""
    echo "  # Training without evaluation (fastest)"
    echo "  $0 --disable_fast_eval --disable_full_eval"
    echo ""
    echo "  # Training with only final evaluation"
    echo "  $0 --disable_fast_eval"
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        --device)
            DEVICE="$2"
            shift 2
            ;;
        --save_every_n_steps)
            SAVE_EVERY_N_STEPS="$2"
            shift 2
            ;;
        --rebuild_cache)
            REBUILD_CACHE=true
            shift
            ;;
        --disable_fast_eval)
            ENABLE_FAST_EVAL=false
            shift
            ;;
        --disable_full_eval)
            ENABLE_FULL_EVAL=false
            shift
            ;;
        --help|-h)
            show_usage
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            show_usage
            exit 1
            ;;
    esac
done

# Check if config file exists
if [[ ! -f "$CONFIG" ]]; then
    echo "Error: Configuration file not found: $CONFIG"
    exit 1
fi

echo "🚀 Starting BitMar Training with Evaluation"
echo "  • Configuration: $CONFIG"
echo "  • Device: $DEVICE"
echo "  • Fast evaluation: $ENABLE_FAST_EVAL"
echo "  • Full evaluation: $ENABLE_FULL_EVAL"
if [[ -n "$SAVE_EVERY_N_STEPS" ]]; then
    echo "  • Step-based checkpoints: every $SAVE_EVERY_N_STEPS steps"
fi
echo ""

# Build the Python command
CMD="python3 train_100M_tokens.py --config $CONFIG --device $DEVICE"

if [[ "$REBUILD_CACHE" == true ]]; then
    CMD="$CMD --rebuild_cache"
fi

if [[ -n "$SAVE_EVERY_N_STEPS" ]]; then
    CMD="$CMD --save_every_n_steps $SAVE_EVERY_N_STEPS"
fi

# Export evaluation flags as environment variables for the Python script to pick up
export BITMAR_ENABLE_FAST_EVAL="$ENABLE_FAST_EVAL"
export BITMAR_ENABLE_FULL_EVAL="$ENABLE_FULL_EVAL"

echo "📝 Running command: $CMD"
echo ""

# Run the training
eval $CMD

echo ""
echo "🎉 Training completed!"

# Show evaluation results if they exist
if [[ -d "evaluation_results" ]]; then
    echo ""
    echo "📊 Evaluation Results Summary:"

    # Show epoch evaluation results
    if ls evaluation_results/epoch_* >/dev/null 2>&1; then
        echo "  • Epoch evaluations:"
        for epoch_dir in evaluation_results/epoch_*; do
            if [[ -d "$epoch_dir" ]]; then
                epoch_num=$(basename "$epoch_dir" | sed 's/epoch_//')
                echo "    - Epoch $epoch_num: $epoch_dir"
            fi
        done
    fi

    # Show final evaluation results
    if [[ -d "evaluation_results/final" ]]; then
        echo "  • Final evaluation: evaluation_results/final"
    fi

    echo ""
    echo "📂 To view detailed results:"
    echo "  find evaluation_results -name '*.json' | head -5"
fi
