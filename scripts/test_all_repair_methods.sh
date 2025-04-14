#! /bin/bash

ALL_REPAIRD_METHODS=(
    "fail_only"
    "fim"
    "masked"
    "mixed"
    "unlearn"
)
SEED=24

AGENT_PATHS=(
    "saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}_repaired_mode_fail_only.pt"
    "saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}_repaired_mode_fim.pt"
    "saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}_repaired_mode_masked.pt"
    "saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}_repaired_mode_mixed.pt"
    "saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}_repaired_mode_unlearn.pt"
)
AGENT_1_PATH="saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}.pt"

# create logs directory if it doesn't exist
mkdir -p logs
for AGENT_PATH in "${AGENT_PATHS[@]}"; do
    rm missed_*.txt
    rm results.csv
    LOG_FILE="logs/$(basename "$AGENT_PATH" .pt)_log.txt"
    echo "Testing agent: $AGENT_PATH" | tee "$LOG_FILE"
    python benchmark.py \
        --agent_0_path "$AGENT_PATH" \
        --agent_1_path "$AGENT_1_PATH"
    echo "repaired controller missed:" | tee -a "$LOG_FILE"
    cat missed_0.txt | wc -l | tee -a "$LOG_FILE"
    echo "poisoned controller missed:" | tee -a "$LOG_FILE"
    cat missed_1.txt | wc -l | tee -a "$LOG_FILE"
    echo "controller 2 missed:" | tee -a "$LOG_FILE"
    cat missed_2.txt | wc -l | tee -a "$LOG_FILE"
    echo "controller 3 missed:" | tee -a "$LOG_FILE"
    cat missed_3.txt | wc -l | tee -a "$LOG_FILE"
    echo "controller 4 missed:" | tee -a "$LOG_FILE"
    cat missed_4.txt | wc -l | tee -a "$LOG_FILE"
done