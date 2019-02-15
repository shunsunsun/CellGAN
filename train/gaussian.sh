#!/bin/bash

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0

echo "Training..."
python -m cellgan.experiments.gaussian.run_gaussian \
    --disc_learning_rate 2e-4 \
    --gen_learning_rate 5e-4 \
    --experts 10 \
    --num_cell_cnns 30 \
    --num_iter 10000
