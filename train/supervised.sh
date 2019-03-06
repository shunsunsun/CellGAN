#!/bin/bash

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0

echo "Training classifier on CellGAN model..."
python -m cellgan.supervised.run_supervised \
    --inhibitor AKTi \
    --learning_rate 1e-5 \
    --num_iter 60000 \
    --print_every_n 500 \
    --cofactor 5 \
    --subpopulation_limit 30 \
    --num_filters 50

