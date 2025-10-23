#!/bin/bash
# Quick start script for 8-GPU baseline training
# This launches training in a tmux session for easy monitoring

set -e

SESSION_NAME="deltanet-baseline-8gpu"

echo "============================================"
echo "DeltaNet 340M Baseline - 8 GPU Training"
echo "============================================"
echo ""
echo "This will launch training in tmux session: ${SESSION_NAME}"
echo ""
echo "To monitor progress:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "To detach from tmux:"
echo "  Press Ctrl+B, then press D"
echo ""
echo "To stop training:"
echo "  tmux kill-session -t ${SESSION_NAME}"
echo ""
echo "============================================"
echo ""

# Kill existing session if any
tmux kill-session -t ${SESSION_NAME} 2>/dev/null || true

# Get script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Launch training in tmux
cd ${SCRIPT_DIR}
tmux new-session -d -s ${SESSION_NAME} "bash train_baseline_8gpu.sh; echo ''; echo 'Training finished. Press any key to close.'; read"

echo "✓ Training launched in tmux session: ${SESSION_NAME}"
echo ""
echo "Attach to monitor:"
echo "  tmux attach -t ${SESSION_NAME}"
echo ""
echo "View logs in real-time:"
echo "  tmux capture-pane -t ${SESSION_NAME} -p"
echo ""
echo "Check wandb dashboard:"
echo "  https://wandb.ai/"
echo ""
echo "============================================"



