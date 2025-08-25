# Message-Passing Δ-ML with Electronically Informed Descriptors

## Authors
- Adam Coxson
- Omer Omar
- Marcos del Cueto
- Alessandro Troisi

## Abstract
We present a machine learning approach capable of enhancing the accuracy of semiempirical excited-state energy calculations (ZINDO) with respect to reference TDDFT results. Our model incurs negligible additional computational cost and improves the correlation between ZINDO and TDDFT excitation energies from 0.75 to 0.95, enabling efficient screening of large molecular datasets with Δ-ML corrected ZINDO calculations.

Using a dataset of 10,000 organic π-conjugated molecules with energies computed at both the ZINDO and M06-2X/3-21G* TDDFT levels, we trained a model to learn the systematic errors of the low-level method and correct it toward high-level accuracy. Critical to the model’s performance is an AttentiveFP message-passing neural network augmented with electronic information from ZINDO (e.g., particle–hole densities from the transition density matrix). We also explore the utility of the Morgan fingerprint and introduce a novel descriptor for molecular electronic structure based on the radial distribution function weighted by molecular orbital coefficients. Finally, we discuss potential applications of this Δ-ML approach in virtual screening.

## Usage
To run the pretrained model examples, create a suitable environment (See requirements), then open and run test_model.py. 

**Note:** Tutorials for model training will be uploaded later.

## Citation
If you use this code or build upon our work, please cite our paper using the following BibTeX entry:

```bibtex
@article{coxson2025predicting,
  title={Predicting S1 TDDFT energies from ZINDO calculations using Message-Passing Delta-ML with electronically-informed descriptors},
  author={Coxson, Adam and Omar, Omer and Del Cueto, Marcos and Troisi, Alessandro},
  year={2025},
  journal={ChemRxiv},
  doi={10.26434/chemrxiv-2025-gln8z-v2}
}****
```

## Requirements
This project was developed and tested on **Python 3.10+** using a minimal Anaconda environment on Windows.  
The following packages are required:

- [pandas](https://pandas.pydata.org/)  
- [numpy](https://numpy.org/)  
- [scipy](https://scipy.org/)  
- [matplotlib](https://matplotlib.org/)  
- [scikit-learn](https://scikit-learn.org/stable/)  
- [PyTorch](https://pytorch.org/)  
- [torchvision](https://pytorch.org/vision/stable/index.html)  
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/)  
- [xyz2mol](https://github.com/jensengroup/xyz2mol)  
- [RDKit](https://www.rdkit.org/)  

### Installation

To create a working environment, run:

```bash
conda create --name dml-minimal python=3.10
conda activate dml-minimal

conda install -c conda-forge pandas numpy matplotlib xyz2mol rdkit
conda install -c anaconda scipy scikit-learn

pip install torch torchvision torch_geometric
```
