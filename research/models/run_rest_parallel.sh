#!/bin/zsh
# Parallel completion of the retraining chain after the main 5-fold CV run.
# DAG: run 1 (main CV) is done → these four streams are independent:
#   A) fixed-split gnn_best          (needs nothing further)
#   B) ablation GAT 2-layer          (own out-dir)
#   C) ablation GCN 3-layer          (own out-dir)
#   D) GA weight tuning → test eval  (needs only run-1 fold checkpoints)
# OMP_NUM_THREADS caps keep 4 concurrent PyTorch processes from
# oversubscribing the 10 cores.
cd "$(dirname "$0")"

export OMP_NUM_THREADS=3
python3 -u train_gnn.py --cv-folds 0 > retrain_fixed.log 2>&1 &
PID_A=$!

python3 -u train_gnn.py --cv-folds 5 --layers 2 \
  --out-dir checkpoints_ablation/gat_2layer > retrain_abl_gat2l.log 2>&1 &
PID_B=$!

python3 -u train_gnn.py --cv-folds 5 --conv gcn \
  --out-dir checkpoints_ablation/gcn_3layer > retrain_abl_gcn.log 2>&1 &
PID_C=$!

{
  export OMP_NUM_THREADS=2
  python3 -u run_ga_ensemble.py --oof > retrain_ga.log 2>&1 &&
  python3 -u evaluate_test_set.py --show-rankings > retrain_eval.log 2>&1
} &
PID_D=$!

wait $PID_A; echo "stream A (fixed-split) done: $?"
wait $PID_B; echo "stream B (GAT 2L) done: $?"
wait $PID_C; echo "stream C (GCN) done: $?"
wait $PID_D; echo "stream D (GA+eval) done: $?"
echo "=== PARALLEL REST ALL DONE ==="
