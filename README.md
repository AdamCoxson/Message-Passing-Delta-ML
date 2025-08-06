Predicting S<sub>1</sub> TDDFT energies from ZINDO calculations using Message-Passing Δ-ML with electronically-informed descriptors
Authors
Adam Coxson
Omer Omar
Marcos del Cueto
Alessandro Troisi
Abstract
We present a machine learning approach capable of enhancing the accuracy of semiempirical excited-state energy calculations (ZINDO) with respect to reference TDDFT results. Our model incurs negligible additional computational cost and improves the correlation between ZINDO and TDDFT excitation energies from 0.75 to 0.95, enabling efficient screening of large molecular datasets with Δ-ML corrected ZINDO calculations. Using a dataset of 10,000 organic π-conjugated molecules with energies computed at both the ZINDO and M06-2X/3-21G* TDDFT levels, we trained a model to learn the systematic errors of the low-level method and correct it toward high-level accuracy. Critical to the model’s performance is an AttentiveFP message-passing neural network augmented with electronic information from ZINDO (e.g., particle–hole densities from the transition density matrix). We also explore the utility of the Morgan fingerprint and introduce a novel descriptor for molecular electronic structure based on the radial distribution function weighted by molecular orbital coefficients. Finally, we discuss potential applications of this Δ-ML approach in virtual screening. Note: Tutorials and scripts for data processing and model training will be uploaded later, along with examples demonstrating the use of this repository.
Citation

If you use this code or build upon our work, please cite our paper. For example, you can use the following BibTeX entry:

@article{Coxson2025DeltaML,
  title   = {Predicting $S_1$ TDDFT energies from ZINDO calculations using Message-Passing $\Delta$-ML with electronically-informed descriptors},
  author  = {Coxson, Adam and Omar, Omer and del Cueto, Marcos and Troisi, Alessandro},
  year    = {2025},
  note    = {Preprint (submitted)}
}
