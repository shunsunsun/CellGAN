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
	bash train/gmm.sh

gaussian:
	bash train/gaussian.sh

bodenmiller:
	bash train/bodenmiller.sh

