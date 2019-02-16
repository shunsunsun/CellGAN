#!/bin/bash

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0

echo "Training Gaussian Mixture Model on real data..."
python -m cellgan.experiments.baselines.GMM.run_GMM \
    --inhibitor AKTi \
    --strength A02 \
    --learning_rate 5e-4 \
    --experts 20 \
    --num_cell_cnns 50 \
    --num_iter 6000 \
    --real_vs_expert \
    --each_subpop
