# -*- coding: utf-8 -*-
"""
Created on Thu Jun  5 15:27:29 2025

@author: adamc
"""

from typing import List, Optional
from rdkit import Chem
import torch
from torch_geometric.data import Data,Dataset
from rdkit.Geometry import Point3D
from xyz2mol import xyz2mol
import numpy as np
import traceback


class MoleculeDataset(Dataset):
    """
    Custom PyTorch Geometric Dataset for molecular data.
    """
    def __init__(self,
                 dataframe,
                 extra_feature_headers: Optional[List[str]] = None
                 ):
        """
        Args:
            smiles_list (list): List of SMILES strings for each molecule
            charges (np.ndarray): Array of atomic charges (9056 x N)
            coordinates (np.ndarray): Array of 3D coordinates (9056 x N x 3)
            morgan_fps (np.ndarray): Array of Morgan fingerprints (9056 x M)
            zindo_energies (np.ndarray): Array of Zindo energies (9056 x 1)
            mulliken_charges (np.ndarray, optional): Array of Mulliken charges (9056 x N)
            particle_densities (np.ndarray, optional): Array of weighted particle densities (9056 x N)
            transform (callable, optional): Transform to be applied on each data object
            pre_transform (callable, optional): Transform to be applied on each data object before saving
        """
        super(MoleculeDataset, self).__init__()
        self.dataframe=dataframe
        self.extra_feature_headers = extra_feature_headers or None
        
        self.symbols = [
            'H', 'B', 'C', 'N', 'O', 'F', 'Si', 'P', 'S', 'Cl', 'As', 'Se', 'Br',
            'Te', 'I', 'At', 'other'
        ]
    
        self.hybridizations = [
            Chem.rdchem.HybridizationType.S,
            Chem.rdchem.HybridizationType.SP,
            Chem.rdchem.HybridizationType.SP2,
            Chem.rdchem.HybridizationType.SP3,
            Chem.rdchem.HybridizationType.SP3D,
            Chem.rdchem.HybridizationType.SP3D2,
            'other',
        ]
    
        self.stereos = [
            Chem.rdchem.BondStereo.STEREONONE,
            Chem.rdchem.BondStereo.STEREOANY,
            Chem.rdchem.BondStereo.STEREOZ,
            Chem.rdchem.BondStereo.STEREOE,
        ]
        
        # Process molecules to create graph data
        self.data_list = self.form_mol_dataset(self.dataframe)
        
        
        print(f"Successfully processed {len(self.data_list)} molecules out of {len(dataframe)}")
    
    def get_mol_features(self, mol, extra_features=None):
        atom_coords = mol.GetConformer().GetPositions()
        xs = []
    
        for idx, atom in enumerate(mol.GetAtoms()):
            symbol = [0.] * len(self.symbols)
            try:
                symbol_idx = self.symbols.index(atom.GetSymbol())
            except ValueError:
                symbol_idx = self.symbols.index("other")
            symbol[symbol_idx] = 1.
    
            degree = [0.] * 6
            degree[min(atom.GetDegree(), 5)] = 1.
            formal_charge = atom.GetFormalCharge()
            radical_electrons = atom.GetNumRadicalElectrons()
            hybridization = [0.] * len(self.hybridizations)
            hybrid = atom.GetHybridization()
            try:
                if atom.GetSymbol() == 'H' and hybrid == Chem.rdchem.HybridizationType.UNSPECIFIED:
                    hybridization[0] = 1.
                elif atom.GetSymbol() != 'H' and hybrid == Chem.rdchem.HybridizationType.UNSPECIFIED:
                    hybridization[-1] = 1. 
                else:
                    hybridization[self.hybridizations.index(hybrid)] = 1.
            except:
                print(hybrid, atom.GetSymbol())

    
            aromaticity = 1. if atom.GetIsAromatic() else 0.
    
            hydrogens = [0.] * 5
            hydrogens[min(atom.GetTotalNumHs(), 4)] = 1.
    
            chirality = 1. if atom.HasProp('_ChiralityPossible') else 0.
            chirality_type = [0.] * 2
            if atom.HasProp('_CIPCode'):
                try:
                    chirality_type[['R', 'S'].index(atom.GetProp('_CIPCode'))] = 1.
                except ValueError:
                    pass
    
            feature_vector = (
                symbol + degree + [formal_charge] +
                [radical_electrons] + hybridization + [aromaticity] +
                hydrogens + [chirality] + chirality_type +
                atom_coords[idx].tolist()
            )
    
            if extra_features is not None:
                feature_vector += list(extra_features[idx])
                
            xs.append(torch.tensor(feature_vector, dtype=torch.float))
    
        x = torch.stack(xs, dim=0)
    
        # --- Edge features ---
        edge_indices = []
        edge_attrs = []
        for bond in mol.GetBonds():
            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
            edge_indices += [[i, j], [j, i]]
    
            bond_type = bond.GetBondType()
            single = 1. if bond_type == Chem.rdchem.BondType.SINGLE else 0.
            double = 1. if bond_type == Chem.rdchem.BondType.DOUBLE else 0.
            triple = 1. if bond_type == Chem.rdchem.BondType.TRIPLE else 0.
            aromatic = 1. if bond_type == Chem.rdchem.BondType.AROMATIC else 0.
            conjugation = 1. if bond.GetIsConjugated() else 0.
            ring = 1. if bond.IsInRing() else 0.
    
            stereo = [0.] * 4
            try:
                stereo[self.stereos.index(bond.GetStereo())] = 1.
            except ValueError:
                pass
    
            edge_attr = torch.tensor([single, double, triple, aromatic, conjugation, ring] + stereo)
            edge_attrs += [edge_attr, edge_attr]
    
        if len(edge_attrs) == 0:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_attr = torch.zeros((0, 10), dtype=torch.float)
        else:
            edge_index = torch.tensor(edge_indices).t().contiguous()
            edge_attr = torch.stack(edge_attrs, dim=0)
    
        return x, edge_index, edge_attr

    
   
    
    def form_mol_dataset(self, dataframe):
        data_list=[]
        for i, row in dataframe.iterrows():
            if i in np.arange(500,9500,500):
                print(f"Row {i} - SMILES: {row['smiles']}")

            if self.extra_feature_headers is not None:
                feature_columns = [row[hdr] for hdr in self.extra_feature_headers]
                extra_features_list = list(zip(*feature_columns))
            else:
                extra_features_list=None
            if type(row['elements'][0]) != int: 
                int_type_atoms = getattr(row['elements'], "tolist", lambda: int)()
            else:
                int_type_atoms=row['elements']
            try:
                mol_list = xyz2mol(atoms=int_type_atoms, coordinates=row['coords'])
                if not mol_list or mol_list[0] is None:
                    print(f"❌ Invalid mol at row {i}: {row['zinc_id']}")
                    continue
                mol = mol_list[0]
            except Exception as e:
                print(f"❌ Exception on molecule {i}: {e}")
                continue
            mol = Chem.AddHs(mol)  # Add explicit Hs (important for correct atom indexing)
            
            rdkit_elements = [atom.GetAtomicNum() for atom in mol.GetAtoms()]

            # Confirm order
            for j, (rdkit_elem, coord_elem) in enumerate(zip(rdkit_elements, row['elements'])):
                assert rdkit_elem == coord_elem, f"Mismatch at atom {i}: RDKit has {rdkit_elem}, data has {coord_elem}"
            
            # Generate a conformer with the same number of atoms
            conf = Chem.Conformer(mol.GetNumAtoms())
            # Assign your coordinates
            for j, (x, y, z) in enumerate(row['coords']):
                conf.SetAtomPosition(j, Point3D(x, y, z))
            
            # Set the conformer to the molecule
            mol.RemoveAllConformers()
            mol.AddConformer(conf)
            pos = torch.tensor(row['coords'], dtype=torch.float)
            try:
                x, edge_index, edge_attr = self.get_mol_features(mol,extra_features=extra_features_list)
            except Exception as error:
                print(i, row)
                print("An exception occurred:", error)
                print("Exception type:", type(error).__name__)
                print(traceback.format_exc())
            try:
                mol_id=row['zinc_id']
            except:
                mol_id=row['mol_id']
            data = Data(
                x=x,
                edge_index=edge_index,
                edge_attr=edge_attr,
                e_zindo=torch.tensor([row['e_zindo']], dtype=torch.float32).view(1),
                y=torch.tensor([row['e_tddft']], dtype=torch.float32).view(1),
                z=row['elements'],
                pos=pos,
                smiles=row['smiles'],
                morganfp=torch.tensor(row['morgan_fp'], dtype=torch.float),
                rdf_homo=torch.tensor(row['rdf_homo'], dtype=torch.float32),
                rdf_lumo=torch.tensor(row['rdf_lumo'], dtype=torch.float32),
                name=mol_id,
                idx=i,
            )
            data_list.append(data)
        return data_list
    
    def len(self):
        return len(self.data_list)
    
    def get(self, idx):
        return self.data_list[idx]
    
    