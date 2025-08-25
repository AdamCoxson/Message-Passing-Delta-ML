# Message-Passing Δ-ML with Electronically Informed Descriptors

Code repository for [ChemRxiv paper](https://chemrxiv.org/engage/chemrxiv/article-details/68a7528423be8e43d65717c1) on:

"Predicting S1 TDDFT energies from ZINDO calculations using Message-Passing Delta-ML with electronically-informed descriptors"
  
## Authors
- Adam Coxson
- Omer Omar
- Marcos del Cueto
- Alessandro Troisi
  
**Corresponding Authour:** adamcoxson1@gmail.com

Any problems or questions then please get in touch by opening an issue if related to the code, or for more general enquiries contact my email.

## Abstract
We present a machine learning approach capable of enhancing the accuracy of semiempirical excited-state energy calculations (ZINDO) with respect to reference TDDFT results. Our model incurs negligible additional computational cost and improves the correlation between ZINDO and TDDFT excitation energies from 0.75 to 0.95, enabling efficient screening of large molecular datasets with Δ-ML corrected ZINDO calculations.

Using a dataset of 10,000 organic π-conjugated molecules with energies computed at both the ZINDO and M06-2X/3-21G* TDDFT levels, we trained a model to learn the systematic errors of the low-level method and correct it toward high-level accuracy. Critical to the model’s performance is an AttentiveFP message-passing neural network augmented with electronic information from ZINDO (e.g., particle–hole densities from the transition density matrix). We also explore the utility of the Morgan fingerprint and introduce a novel descriptor for molecular electronic structure based on the radial distribution function weighted by molecular orbital coefficients. Finally, we discuss potential applications of this Δ-ML approach in virtual screening.

## Usage
To apply the pretrained models to the test and per-fold validation sets, create a suitable environment (See requirements), then open and run test_model.py. You can see an example of the console output in "Example_console_output_from_test_models.txt"
  
Directories:
- data: Preprocessed dataset class objects for 407 molecule testing set and the 9050 molecule training set**, as well as the train-validation fold indices. These are saved as .pkl files (0.5 GB uncompressed).
- pretrained_models: Contains 3 selected models from the preprint: Dense(MPNN(DA, EA), E_zindo), MPNN(DA, EA), and MPNN(DA). (0.8 GB uncompressed)
- saves: Empty folder where plots and csv are saved to after running test_model.py.
- 5_mol_ZINDO_mwfn_examples: Examples of Gaussian-16 and Multi-Wavefunction log files used to create the datasets. Contains bash scripts to run wvfn calculations.(5 MB)

**Note:** 
- Tutorials for model training will be uploaded later.
- **Training data (150 Mb Zip file) is available from the [release page](https://github.com/AdamCoxson/Message-Passing-Delta-ML/releases) or by clicking this link [train_data.zip](https://github.com/AdamCoxson/Message-Passing-Delta-ML/releases/download/v0.1.0/pre-processed_trainset_9050_mols.7z)

## Citation
If you use this code or build upon our work, please cite our paper using the following BibTeX entry:

```bibtex
@article{coxson2025predicting,
  title={Predicting S1 TDDFT energies from ZINDO calculations using Message-Passing Delta-ML with electronically-informed descriptors},
  author={Coxson, Adam and Omar, Omer and Del Cueto, Marcos and Troisi, Alessandro},
  year={2025},
  journal={ChemRxiv},
  doi={10.26434/chemrxiv-2025-gln8z-v2}
}
```

## Requirements
This project was developed and tested on **Python 3.10+** using an Anaconda environment on Windows.  
The following packages are required:

- [pandas](https://pandas.pydata.org/docs/getting_started/install.html)  
- [numpy](https://numpy.org/install/)
- [scipy](https://scipy.org/install/)  
- [matplotlib](https://matplotlib.org/stable/users/getting_started/)  
- [scikit-learn](https://scikit-learn.org/stable/install.html)  
- [PyTorch](https://pytorch.org/get-started/locally/)  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html)  
- [xyz2mol](https://github.com/jensengroup/xyz2mol)  
- [RDKit](https://github.com/rdkit/rdkit)

### Installation (via anaconda)

To create a minimal working environment on an anaconda distribution, run:

```bash
conda create --name dml-minimal python=3.10
conda activate dml-minimal
conda install -c conda-forge pandas numpy matplotlib xyz2mol rdkit
conda install -c anaconda scipy scikit-learn
pip install torch torchvision torch_geometric
```
****
