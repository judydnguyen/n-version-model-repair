#! /bin/bash

rm missed_*
rm results.csv

python benchmark.py \
    --agent_0_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234_repaired_mode_unlearn.pt \
    --agent_1_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_1234.pt

echo "repaired controller missed:"
cat missed_0.txt | wc -l
echo "poisoned controller missed:"
cat missed_1.txt | wc -l
echo "controller 2 missed:"
cat missed_2.txt | wc -l
echo "controller 3 missed:"
cat missed_3.txt | wc -l
echo "controller 4 missed:"
cat missed_4.txt | wc -l

