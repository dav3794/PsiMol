# PsiMol

## Overview

This project is a lightweight Python plugin for [psi4 package](https://psicode.org) for automated reading, parsing, conversion and basic molecular operations on chemical formats, including xyz, mol, cif and SMILES.

The tool enables some standalone transformations of molecules, including adding explicit hydrogens, mutating atoms/bonds and getting basical chemical properties (like molar mass or total charge), and parsing between the above file formats. It also allows for interactive molecule visualisations in Jupyter notebooks.

## Installation

To use the tool, you need to have Conda installed (Miniconda is completely sufficient). Then simply run:

```bash
cd PsiMol
conda env create -f environment.yml
conda activate psimol-env
pip install .
```

## Usage

### Python package

PsiMol can be simply imported within Python scripts. Example:

```python
import psimol

molecule = psimol.Molecule.from_xyz('file_path')
molecule.add_hydrogens()
print(molecule.molar_mass)
```

For more examples of PsiMol usage, see *psimol/tutorial.ipynb* notebook.

### CLI

PsiMol is also available as a command-line tool with multiple functionalities. Example:

```bash
psimol add_hydrogens -i file -o file_with_H --format xyz
```
