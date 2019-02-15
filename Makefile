# Formatting and checks

clean:
	yapf -i -r cellgan/

check:
	flake8

# CPU scripts: Test model without errors

gmm-cpu:
	bash train-cpu/gmm.sh

gaussian-cpu:
	bash train-cpu/gaussian.sh

bodenmiller-cpu:
	bash train-cpu/bodenmiller.sh

# GPU scripts

gmm:
	bsub -n 4 -N -W 4:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/gmm.sh

gaussian:
	bsub -n 8 -N -W 24:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/gaussian.sh

bodenmiller:
	bsub -n 8 -N -W 30:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/bodenmiller.sh

