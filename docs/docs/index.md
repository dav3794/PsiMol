# PsiMol documentation


## Overview

PsiMol is a lightweight Python plugin for [Psi4](https://psicode.org) for the automated reading, parsing and conversion of various chemical formats, as well as basic operations on molecules.

PsiMol is available under the [MIT License](https://opensource.org/license/mit).


### Features

* Deserialization from chemical formats: XYZ, MOL, SDF, CIF (including mmCIF), SMILES
* Serialization to chemical formats: XYZ, MOL
* Interoperation with Psi4
* Molecule visualizations in py3Dmol
* Basic operations on molecules:
    * Molar mass calculation
    * Charge calculation
    * Adding and removing atoms
    * Adding explicit hydrogens
    * Geometry optimization
    * Vibrational frequency calculation
    * Single-point energy calculation


## Installation

Conda (or Miniconda) is required to use PsiMol.
Within Conda's environment, run the following commands in Bash:

```bash
cd PsiMol
conda env create -f environment.yml
conda activate psimol-env
pip install .
```

This will install all of PsiMol's dependencies and create an environment for it.


## Usage

PsiMol is available as a [Python module](python/intro.md), and also provides a [CLI tool](cli/intro.md).

There's also a brief tutorial for PsiMol available in the form of a Jupyter notebook.

