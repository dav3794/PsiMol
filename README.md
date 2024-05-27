## PsiMol

by *Dawid Uchal, Maciej Bielecki and Paweł Nagórko*

## Overview

This project is a lightweight Python plugin for [psi4 package](https://psicode.org) for automated reading, conversion and basic molecular operations on chemical formats, including xyz, mol and (most importantly) SMILES.

#### Future features

The tool will enable some standalone transformations of molecules, including adding explicit hydrogens, mutating atoms/bonds and getting basical chemical properties (like molar mass or total charge), and parsing between the above file formats.

Additional, minor task would include implementing visualizations of obtained molecules.

## Installation

To use the tool, you need to have Conda installed (Miniconda is completely sufficient). Then simply run:

```bash
cd PsiMol
conda env create -f environment.yml
conda activate psimol-env
pip install -e .
```

## Usage

### Python package

PsiMol can be simply imported within Python scripts. Example:

```python
import psimol

molecule = psimol.Molecule.from_xyz('file_path')
molecule = molecule.add_hydrogens()
print(molecule.molar_mass)
```

### CLI

PsiMol is also available as a command-line tool with multiple functionalities. Example:

```bash
psimol add_hydrogens -i file -o file_with_H --format xyz
```
