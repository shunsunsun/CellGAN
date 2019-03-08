
# Some defaults (TODO: Add these for all methods)
INHIB=AKTi
STREN=A02
RUNS=10

# Format and checks
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

phenograph-cpu:
	bash train-cpu/phenograph.sh

supervised-cpu:
	bash train-cpu/supervised.sh

flowsom:
	Rscript cellgan/experiments/baselines/FlowSOM/run_flowSOM.R $(INHIB) $(STREN) $(RUNS)
	python -m cellgan.experiments.baselines.FlowSOM.evaluate_flowsom --inhibitor $(INHIB) --strength $(STREN) --nruns $(RUNS)

# GPU scripts
gmm:
	bsub -n 4 -N -W 4:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/gmm.sh

gaussian:
	bsub -n 8 -N -W 24:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/gaussian.sh

bodenmiller:
	bsub -n 8 -N -W 30:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/bodenmiller.sh

phenograph:
	bsub -n 8 -N -W 4:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/phenograph.sh

supervised:
	bsub -n 8 -N -W 4:30 -R "rusage[mem=2048,ngpus_excl_p=1]" ./train/supervised.sh

# Updating experiments

update-cellgan:
	python -m cellgan.update --method cellgan --inhibitor AKTi

update-gmm:
	python -m cellgan.update --method baseline --baseline_method GMM --inhibitor AKTi

