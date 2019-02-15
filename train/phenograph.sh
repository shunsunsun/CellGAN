#!/bin/bash

module load python_gpu/3.6.4 cuda/9.0.176 cudnn/7.0

echo "Training PhenoGraph on real data..."
python -m cellgan.experiments.bodenmiller.run_bodenmiller \
	--inhibitor AKTi \
	--strength A02 \
    --n_f_measure_rare 250 \
    --sub_limit 30 \
    --cofactor 5 \
