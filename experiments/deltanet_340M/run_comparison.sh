#!/bin/bash

# Convenience script to run both training experiments in tmux sessions
# This allows monitoring both runs simultaneously

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "============================================"
echo "DeltaNet 340M Comparison Training"
echo "============================================"
echo ""
echo "This script will launch two training runs:"
echo "  1. Baseline DeltaNet (in tmux session: deltanet-baseline)"
echo "  2. DeltaNet with OSLA (in tmux session: deltanet-osla)"
echo ""
echo "You can attach to each session to monitor progress:"
echo "  tmux attach -t deltanet-baseline"
echo "  tmux attach -t deltanet-osla"
echo ""
echo "Press Ctrl+B then D to detach from a tmux session"
echo "============================================"
echo ""

# Check if tmux is available
if ! command -v tmux &> /dev/null; then
    echo "Error: tmux is not installed. Please install it first:"
    echo "  sudo apt-get install tmux"
    exit 1
fi

# Kill existing sessions if they exist
tmux kill-session -t deltanet-baseline 2>/dev/null || true
tmux kill-session -t deltanet-osla 2>/dev/null || true

echo "Starting baseline DeltaNet training in tmux session 'deltanet-baseline'..."
tmux new-session -d -s deltanet-baseline "cd ${SCRIPT_DIR} && bash train_baseline.sh; bash"

echo "Starting OSLA DeltaNet training in tmux session 'deltanet-osla'..."
tmux new-session -d -s deltanet-osla "cd ${SCRIPT_DIR} && bash train_osla.sh; bash"

echo ""
echo "✓ Both training runs have been launched!"
echo ""
echo "To monitor the runs:"
echo "  tmux attach -t deltanet-baseline  # Attach to baseline run"
echo "  tmux attach -t deltanet-osla      # Attach to OSLA run"
echo ""
echo "To list all sessions:"
echo "  tmux ls"
echo ""
echo "To detach from a session:"
echo "  Press Ctrl+B, then press D"
echo ""
echo "To kill a session:"
echo "  tmux kill-session -t <session-name>"
echo ""
echo "Training logs will be saved to:"
echo "  ${SCRIPT_DIR}/outputs/deltanet_340M_baseline/"
echo "  ${SCRIPT_DIR}/outputs/deltanet_340M_osla/"
echo ""
echo "Monitor W&B dashboard at: https://wandb.ai/"
echo "============================================"



