# !/bin/bash

# This script trains the agent on the with different seeds.
# Prolly, you may want to change the attack budgets of the agents a little bit.

ATK_BUDGETs=(100 200 200 300 400)
SEEDS=(1234 5678 91011 121314 151617)

for i in ${!ATK_BUDGETs[@]};
do
    python train_attack.py --seed ${SEEDS[$i]} --attack_budget ${ATK_BUDGETs[$i]} --starting_attack_eps 500
done
