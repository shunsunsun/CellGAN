#!/bin/bash

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0

echo "Preparing files for training..."
python -m cellgan.experiments.bodenmiller.get_bodenmiller_files --regex A02
echo "Files prepared."

echo "Training Gaussian Mixture Model on real data..."
python -m cellgan.experiments.bodenmiller.run_bodenmiller.py \
    --learning_rate 5e-4 \
    --experts 20 \
    --num_cell_cnns 50 \
    --num_iter 6000
echo "Training complete."
