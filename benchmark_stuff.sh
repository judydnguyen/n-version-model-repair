#! /bin/bash

rm missed_*
rm results.csv

python benchmark.py

echo "repaired controller missed (out of 200):"
cat missed_0.txt | wc -l
echo "poisoned controller missed (out of 200):"l
cat missed_1.txt | wc -l
