SEED=24
python repair.py --fim_reg 10 --mixed_ratio 0.5 --repair_mode "unlearn" --old_ckpt_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}.pt \
    --fail_test_path new_failing_cases.csv \
    --pass_test_path new_passing_cases.csv && \
python repair.py --fim_reg 10 --mixed_ratio 0.5 --repair_mode "fail_only" --old_ckpt_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}.pt \
    --fail_test_path new_failing_cases.csv \
    --pass_test_path new_passing_cases.csv && \
python repair.py --fim_reg 10 --mixed_ratio 0.5 --repair_mode "mixed" --old_ckpt_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}.pt \
    --fail_test_path new_failing_cases.csv \
    --pass_test_path new_passing_cases.csv && \
python repair.py --fim_reg 10 --mixed_ratio 0.5 --repair_mode "masked" --old_ckpt_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}.pt \
    --fail_test_path new_failing_cases.csv \
    --pass_test_path new_passing_cases.csv && \
python repair.py --fim_reg 10 --mixed_ratio 0.5 --repair_mode "fim" --old_ckpt_path saved_ckpts/cartpole_reinforce_weights_attacked_seed_${SEED}.pt \
    --fail_test_path new_failing_cases.csv \
    --pass_test_path new_passing_cases.csv
