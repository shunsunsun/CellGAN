# CellGan

CellGan is a Generative Adversarial Network that aims to learn the cellular heterogeneity from flow cytometry data in an interpretable manner and use it for defining subpopulations. 

## Datasets 

* Mixture of Gaussian - For proof-of-concept testing
* Bodenmiller Mass Cytometry - More information can be found [here](https://www.nature.com/articles/nbt.2317).
(TODO: Add other datasets?)

## Training

NOTE: All commands must be run in a terminal from project root

The following steps are in reference to running jobs on the Leonhard cluster:

* Train a CellGAN model on the Mixture of Gaussian data, run `make gaussian`
* Train a CellGAN model on the Bodenmiller data run `make bodenmiller`

## Baselines

Baseline methods are run for the Bodenmiller Mass Cytometry data. 
The baseline methods we compare to include (All commands must be run in a terminal from project root): 

* [FlowSOM](https://www.ncbi.nlm.nih.gov/pubmed/25573116): Train FlowSOM by running `make flowsom` (TODO)
* GMM: Train Gaussian Mixture Model by running `make gmm`
* [XShift](https://www.nature.com/articles/nmeth.3863): Train XShift by running `make xshift` (TODO)
* [PhenoGraph](https://www.cell.com/cell/fulltext/S0092-8674(15)00637-6): Train PhenoGraph by running `make pheno`(TODO)

## Third Party Code

### FlowSOM
GitHub: https://github.com/SofieVG/FlowSOM

### XShift
GitHub: https://github.com/nolanlab/vortex/

### PhenoGraph
GitHub: https://github.com/jacoblevine/PhenoGraph
