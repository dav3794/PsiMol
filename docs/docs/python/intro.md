# PsiMol as a Python module


PsiMol can be used as a Python module. To use it in a Python script, first import it:
```python
import psimol
```


## Molecule

Primarily, PsiMol exposes the class `psimol.Molecule`, which is used to describe individual molecules.
A `psimol.Molecule` instance can be created based on various chemical format files and manipulated in various ways.

It should be noted that Psi4 also exposes its own class for molecules (`psi4.Molecule`).
Psi4's and PsiMol's molecule classes are wholly separate (`psimol.Molecule` is implemented from the ground up, not a subclass of `psi4.Molecule`);
however, the `psimol.Molecule` offers conversion to `psi4.Molecule`.


## Bond, Atom

In addition to `psimol.Molecule`, PsiMol exposes two more classes: `psimol.Atom` and `psimol.Bond`, used to describe atoms and bonds between atoms, respectively.
`psimol.Molecule` encapsulates instances of both `psimol.Atom` and `psimol.Bond`, but both of those classes are usable on their own.
In addition, a `psimol.Molecule` instance may provide direct access to its constituent `psimol.Atom` and `psimol.Bond` instances.