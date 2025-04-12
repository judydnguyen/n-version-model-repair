# !/bin/bash

for seed in 10 20 30 40 50 60 70 80 90 100;
do
    python train.py --seed $seed
done