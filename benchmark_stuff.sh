#! /bin/bash

rm missed_*
rm results.csv

python benchmark.py \
    --agent_0_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234_repaired_mode_unlearn.pt \
    --agent_1_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234.pt

echo "repaired controller missed (out of 200):"
cat missed_0.txt | wc -l
echo "poisoned controller missed (out of 200):"
cat missed_1.txt | wc -l
