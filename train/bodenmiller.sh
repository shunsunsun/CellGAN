#!/bin/bash

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0

echo "Preparing files for training..."
python -m cellgan.experiments.bodenmiller.get_bodenmiller_files --regex A02
echo "Files prepared. "

echo "Training CellGAN on real data..."
bsub -n 8 -N -W 30:30 -R "rusage[mem=2048,ngpus_excl_p=1]" python -m cellgan.experiments.bodenmiller.run_bodenmiller \
    --disc_learning_rate 5e-4 \
    --gen_learning_rate 2e-4 \
    --experts 50 \
    --num_cell_cnns 50 \
    --num_iter 10000 \
    --batch_size 64 \
    --noise_size 200 \
    --plot_every_n 500
echo "Training complete."
