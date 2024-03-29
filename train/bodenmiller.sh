#!/bin/bash

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0

echo "Training CellGAN on real data..."
python -m cellgan.experiments.bodenmiller.run_bodenmiller \
    --inhibitor AKTi \
    --strength A06 \
    --disc_learning_rate 5e-4 \
    --gen_learning_rate 2e-4 \
    --experts 30 \
    --num_cell_cnns 60 \
    --num_iter 10000 \
    --batch_size 64 \
    --noise_size 200 \
    --plot_every 500 \
    --each_subpop \
    --real_vs_expert
