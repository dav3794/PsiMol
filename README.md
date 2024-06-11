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

For more examples of PsiMol usage, see `psimol/tutorial.ipynb` notebook.

### CLI

PsiMol is also available as a command-line tool with multiple functionalities.

```
usage: psimol [-h] [-v] [-q] {convert} ...

psimol command line interface

positional arguments:
  {convert}      Command to run.
    convert      Convert molecule file formats

options:
  -h, --help     show this help message and exit
  -v, --verbose  Enable verbose logging.
  -q, --quiet    Disable logging except errors.
```

Converting between files:
```
usage: psimol convert [-h] --input-format {xyz,cif,smiles,mol} --output-format {xyz,mol} -i INPUT -o OUTPUT [--add-hydrogens]

options:
  -h, --help            show this help message and exit
  --input-format {xyz,cif,smiles,mol}
                        Input file format.
  --output-format {xyz,mol}
                        Output file format.
  -i INPUT, --input INPUT
                        Path to the input file.
  -o OUTPUT, --output OUTPUT
                        Path to the output file.
  --add-hydrogens       Add hydrogens to the molecule.
```


## Documentation

The source for PsiMol's documentation, written with [MkDocs](https://www.mkdocs.org/), is available under `docs`.
A pre-generated copy of the documentation is available under `docs/site`.
The pre-generated copy of the documentation should be up to date with the documentation source, but you may rebuild documentation with the following commands:

```bash
pip install mkdocstrings-python
cd PsiMol
mkdocs build --clean --no-directory-urls -f docs/mkdocs.yml
```
