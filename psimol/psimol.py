from __future__ import annotations
import sys
import psi4
import logging
import numpy as np
from typing import Dict, FrozenSet, List, Literal, Tuple, Union

from psimol.utils import get_atom_config, setup_logging

setup_logging(logging.INFO)

class Atom:
    """Class to represent an atom."""

    def __init__(
            self,
            symbol: str,
            name: Union[str, None] = None,
            x: float = 0.0,
            y: float = 0.0,
            z: float = 0.0,
            charge: int = 0,
            ):
        """Itialize the atom.

        Args:
            symbol (str): Atom symbol. Must match supported symbols in periodic table.
            x (float, optional): X-axis coordinate of the atom. Defaults to 0.0.
            y (float, optional): Y-axis coordinate of the atom. Defaults to 0.0.
            z (float, optional): Z-axis coordinate of the atom. Defaults to 0.0.
            charge (int, optional): Charge of an atom. Defaults to 0.
        """
        
        self.symbol: str = symbol
        self.name: str = name 
        self.x: float = x
        self.y: float = y
        self.z: float = z
        self.charge: int = charge

        atom_config = get_atom_config(symbol)
        self._full_name: str = atom_config['name']
        self._mass: float = atom_config['mass']
        self._atomic_number: int = atom_config['atomic_number']
        self._valence: int = atom_config['valence']
        self._covalent_radii: List[float] = atom_config['covalent_radius']
        if 'metalic_radius' in atom_config:
            self._metalic_radius: Union[float, None] = atom_config['metalic_radius']
        else:
            self._metalic_radius: Union[float, None] = None


    def __str__(self):
        atom_name = f'{self.name}; ' if self.name else ''
        return f'{self.symbol} ({atom_name}{self.full_name})'
    
    @property
    def xyz(self) -> np.ndarray:
        """Return the x, y, z coordiantes of the atom.

        Returns:
            np.ndarray: Array with x, y, z coordinates
        """
        return np.array([self.x, self.y, self.z])
    
    @property
    def full_name(self) -> str:
        return self._full_name
    
    @property
    def mass(self) -> float:
        return self._mass
    
    @property
    def atomic_number(self) -> int:
        return self._atomic_number
    
    @property
    def valence(self) -> int:
        return self._valence
    
    @property
    def covalent_radius(self) -> List[float]:
        return self._covalent_radii
    
    @property
    def metalic_radius(self) -> Union[float, None]:
        return self._metalic_radius
    
    @mass.setter
    def mass(self, value: float):
        """Set the mass of the atom. Useful for isotopes.

        Args:
            value (float): Mass of the atom in atomic mass units.
        """
        self._mass = value

    def number_of_possible_bonds(self) -> int:
        """Calculate the maximum number of bonds the atom can have.

        Returns:
            int: Maximum number of bonds of the atom
        """
        return max(0, self._valence - self.charge)


class Bond:
    """Class to represent a bond from a given atom to another."""

    def __init__(
            self,
            first_atom: Atom,
            second_atom: Atom,
            order: Literal[1, 2, 3] = 1,
            aromatic: bool = False
            ):
        """Initialize the bond.

        Args:
            first_atom (Atom): First atom of the bond
            second_atom (Atom): Second atom of the bond
            order (Literal[1, 2, 3], optional): Bond order (single, double or triple). Defaults to 1.
            aromatic (bool, optional): Whether a bond is aromatic. Defaults to False.
        """

        if first_atom == second_atom:
            logging.error('Cannot create a bond of an atom to itself.')
            sys.exit(1)
        
        self._atoms: FrozenSet[Atom] = frozenset((first_atom, second_atom))
        self.order: Literal[1, 2, 3] = order
        self.aromatic: bool = aromatic

        self._bond_length: float = np.sum((first_atom.xyz - second_atom.xyz)**2) # Euclidean distance

    @property
    def atoms(self) -> Tuple[Atom, Atom]:
        return tuple(self._atoms)
    
    @property
    def bond_length(self) -> float:
        """Calculate the bond length as the Euclidean distance
        between atoms.

        Returns:
            float: Bond length.
        """
        first_atom, second_atom = self.atoms
        return np.sum((first_atom.xyz - second_atom.xyz)**2)


class Molecule:
    """Class to represent a molecule."""
    
    def __init__(
            self, 
            name: str,
            atoms: List[Atom] = []
            ):
        """Initialize the molecule.

        Args:
            name (str): Name of the molecule.
        """
        self.name: str = name
        self._atoms: List[Atom] = atoms

        self.create_bonds()

    def create_bonds(self):
        """Creates bonds between atoms basing on
        the differences between their covalent or metallic
        radii (if the distance between atoms is smaller or
        equal to the sum of their radii, the bond is assumed)
        """
        self._bonds: Dict[Atom, List[Bond]] = {} # Dictionary mapping atoms to their bonds
        pass #TODO

    def update_bonds(self):
        """Update the bonds between atoms after adding
        or removing atoms from the molecule."""
        pass #TODO
    
    
    def __str__(self):
        return self.name
    
    def __len__(self):
        return len(self.atoms)
    
    @property
    def molar_mass(self) -> float:
        """Calculate the molar mass of the molecule.

        Returns:
            float: Molar mass of the molecule
        """
        return np.sum(atom.mass for atom in self.atoms)
    
    @property
    def total_charge(self) -> int:
        """Calculate the total charge of the molecule.

        Returns:
            int: Total charge of the molecule
        """
        return np.sum(atom.charge for atom in self.atoms)
    
    @property
    def atoms(self) -> List[Atom]:
        return self._atoms
    
    @property
    def bonds(self) -> Dict[Atom, List[Bond]]:
        return self._bonds
    
    def add_atom(self, atom: Atom):
        """Add an atom to the molecule.

        Args:
            atom (Atom): Atom to be added to the molecule.
        """
        self.atoms.append(atom)
        self.update_bonds()
    
    def remove_atom(self, atom: Atom):
        """Remove an atom from the molecule.

        Args:
            atom (Atom): Atom to be removed from the molecule.
        """
        self.atoms.remove(atom)
        self.update_bonds()
    
    def add_hydrogens(self):
        """Add implicit hydrogens to the molecule"""
        pass #TODO

    @classmethod
    def from_xyz(cls, file_path: str) -> Molecule:
        """Create molecule from .xyz file

        Args:
            file_path (str): Path to the .xyz file
        """
        pass #TODO

    @classmethod
    def from_mol(cls, file_path: str) -> Molecule:
        """Create molecule from .mol file

        Args:
            file_path (str): Path to the .mol file
        """
        pass #TODO

    @classmethod
    def from_smiles(cls, smiles_string: str) -> Molecule:
        """Create molecule from SMILES string representation

        Args:
            smiles_string (str): SMILES string representation of the molecule
        """
        pass #TODO

    def to_xyz(self) -> str:
        """Return molecule representation in .xyz format

        Returns:
            str: String representing molecule in .xyz format
        """
        pass #TODO

    def to_mol(self) -> str:
        """Return molecule representation in .mol format
        
        Returns:
            str: String representing molecule in .mol format
        """
        pass #TODO

    def to_psi4(self) -> psi4.Molecule:
        """Create psi4 molecule object.

        Returns:
            psi4.Molecule: psi4 molecule object
        """
        psi_molecule = psi4.geometry(self.to_xyz())
        return psi_molecule

    def optimize(self, method: str = 'scf/cc-pvdz', reference: str = 'rhf') -> Molecule:
        """Optimize the molecule geometry using psi4.

        Args:
            method (str, optional): Method/basis set to use for geometry 
            optimization. Defaults to 'scf/cc-pvdz'.
            reference (str, optional): Reference wavefunction. Defaults to 'rhf'.

        Returns:
            Molecule: Optimized molecule
        """
        psi_molecule = self.to_psi4()
        psi4.set_options({'reference': reference})
        psi4.optimize(method, molecule=psi_molecule)

        return Molecule.from_xyz(psi_molecule.save_string_xyz())



    