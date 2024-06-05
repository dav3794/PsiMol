from __future__ import annotations
import re
import sys
import psi4
import logging
import numpy as np
from typing import Dict, FrozenSet, List, Literal, Tuple, Union, Set

from .utils import get_atom_config, setup_logging, euclidean_distance

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

        self._configure_atom(symbol)

    def _configure_atom(self, symbol: str):
        """Configure atom properties based on the symbol, using atom properties
        from the configuration file.

        Args:
            symbol (str): Atom symbol.
        """
        atom_config = get_atom_config(symbol)
        self._full_name = atom_config['name']
        self._mass = atom_config['mass']
        self._atomic_number = atom_config['atomic_number']
        self._valence = atom_config['valence']
        self._covalent_radii = atom_config['covalent_radius']
        if 'metallic_radius' in atom_config:
            self._metallic_radius = atom_config['metallic_radius']
        else:
            self._metallic_radius = None

    def __str__(self):
        atom_name = f'{self.name}; ' if self.name is not None else ''
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
    def metallic_radius(self) -> Union[float, None]:
        return self._metallic_radius

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

    def mutate(self, symbol: str):
        """Mutate the atom to another element.

        Args:
            symbol (str): New atom symbol.
        """
        self.symbol = symbol
        self._configure_atom(symbol)


class Bond:
    """Class to represent a bond from a given atom to another."""

    def __init__(
            self,
            first_atom: Atom,
            second_atom: Atom,
            order: Literal[1, 2, 3] = 1,
            aromatic: bool = False,
            metallic: bool = False
    ):
        """Initialize the bond.

        Args:
            first_atom (Atom): First atom of the bond
            second_atom (Atom): Second atom of the bond
            order (Literal[1, 2, 3], optional): Bond order (single, double or triple). Defaults to 1.
            aromatic (bool, optional): Whether a bond is aromatic. Defaults to False.
            metallic (bool, optional): Whether a bond is metallic. Defaults to False.
        """

        if first_atom == second_atom:
            logging.error('Cannot create a bond of an atom to itself.')
            sys.exit(1)

        self._atoms: FrozenSet[Atom] = frozenset((first_atom, second_atom))
        self.order: Literal[1, 2, 3] = order
        self.aromatic: bool = aromatic
        self.metallic: bool = metallic

    def __str__(self):
        if self.metallic:
            bond_sign = '*'
        elif self.aromatic:
            bond_sign = ':'
        elif self.order == 1:
            bond_sign = '-'
        elif self.order == 2:
            bond_sign = '='
        elif self.order == 3:
            bond_sign = '≡'
        return f'{self.atoms[0]} {bond_sign} {self.atoms[1]}'

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
        return euclidean_distance(first_atom.xyz, second_atom.xyz)


class Molecule:
    """Class to represent a molecule."""

    def __init__(
            self,
            name: str,
            atoms: List[Atom],
            bonds: Dict[Atom, List[Bond]] = None
    ):
        """Initialize the molecule.

        Args:
            name (str): Name of the molecule.
            atoms (List[Atom], optional): List of atoms in the molecule.
            bonds (Dict[Atom, List[Bond]], optional): Dictionary mapping atoms to their bonds.
            If not provided, bonds are created based on the covalent radii of atoms and their
            coordinates.
        """
        self.name = name
        self._atoms = atoms
        if not bonds:
            self._bonds = self._create_bonds_from_xyz(atoms)
        else:
            self._bonds = bonds

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

    def print_bonds(self, show_length: bool = False):
        """Print all bonds in the molecule.

        Args:
            show_length (bool, optional): Whether to print bond length. Defaults to False.
        """
        unique_bonds = set()
        for bonds in self.bonds.values():
            for bond in bonds:
                if bond not in unique_bonds:
                    print(f'{bond} {"[%.2f Å]" % bond.bond_length if show_length else ""}')
                    unique_bonds.add(bond)

    def _validate_bonding(self, atom1: Atom, atom2: Atom) -> Union[Bond, None]:
        """Check if the bond between atoms is possible.

        Args:
            atom1 (Atom): First atom of the bond.
            atom2 (Atom): Second atom of the bond.

        Returns:
            Union[Bond, None]: Bond object if the bond is possible, None otherwise.
        """
        bond_order: Union[int, Literal['metallic']] = 0
        distance = euclidean_distance(atom1.xyz, atom2.xyz)

        if atom1.metallic_radius and atom2.metallic_radius:
            metallic_bond_length = atom1.metallic_radius + atom2.metallic_radius
            if np.abs(distance - metallic_bond_length) < 0.2:
                bond_order = 'metallic'

        for order, (cov_rad1, cov_rad2) in enumerate(
                zip(atom1.covalent_radius, atom2.covalent_radius)
        ):
            covalent_bond_length = cov_rad1 + cov_rad2
            if distance - covalent_bond_length < 0.1 / (order + 1):  # tolerance for longer bonds
                bond_order = order + 1
            else:
                break

        if bond_order:  # if bond_order is not 0
            if bond_order == 'metallic':
                return Bond(atom1, atom2, order=1, metallic=True)
            else:
                return Bond(atom1, atom2, order=bond_order)
        return None

    def _find_cycles(self, bonds: Dict[Atom, List[Bond]], aromatic_atoms: Set[str]) -> List[List[Atom]]:
        def dfs(current, start, visited, path):
            visited.add(current)
            path.append(current)
            for bond in bonds[current]:
                next_atom = next(a for a in bond.atoms if a != current)
                if next_atom == start and len(path) > 2:
                    cycles.append(path[:])
                elif next_atom not in visited and next_atom.symbol in aromatic_atoms:
                    dfs(next_atom, start, visited, path)
            path.pop()
            visited.remove(current)

        cycles = []
        for atom in bonds:
            if atom.symbol in aromatic_atoms:
                dfs(atom, atom, set(), [])
        return cycles

    def _is_planar(self, cycle: List[Atom]) -> bool:
        if len(cycle) < 4:
            return True  # Any three points are always planar
        p0 = cycle[0].xyz
        p1 = cycle[1].xyz
        normal = np.cross(p1 - p0, cycle[2].xyz - p0)
        for i in range(3, len(cycle)):
            if not np.isclose(np.dot(normal, cycle[i].xyz - p0), 0.001):
                return False
        return True

    def _check_aromaticity(self, bonds: Dict[Atom, List[Bond]]):
        """Given a dictionary of bonds, check if any of the bonds
        are aromatic. If so, update such bond's 'aromatic' parameter.

        The bonds are aromatic if they are part of planar ring (flat cycle)
        built of atoms from (C, N, O or S).

        Args:
            bonds (Dict[Atom, List[Bond]]): Bonds in the molecule.
        """
        aromatic_atoms = {'C', 'N', 'O', 'S'}

        all_cycles = self._find_cycles(bonds, aromatic_atoms)

        for cycle in all_cycles:
            if self._is_planar(cycle):
                for atom in cycle:
                    for bond in bonds[atom]:
                        if set(bond.atoms).issubset(cycle):
                            bond.aromatic = True

    def _create_bonds_from_xyz(self, atoms: List[Atom]) -> Dict[Atom, List[Bond]]:
        """Creates bonds between atoms basing on
        the differences between their covalent or metallic
        radii (if the distance between atoms is smaller or
        equal to the sum of their radii, the bond is assumed).

        Args:
            atoms (List[Atom]): List of atoms in the molecule.

        Returns:
            Dict[Atom, List[Bond]]: Dictionary mapping atoms to their bonds.
        """
        bonds = {}  # Dictionary mapping atoms to their bonds

        # iterate over all pairs of atoms
        for i, atom1 in enumerate(atoms[:-1]):
            for atom2 in atoms[i + 1:]:

                # check if the bond between atoms is possible
                bond = self._validate_bonding(atom1, atom2)

                # if bond_order is not 0, create a bond
                if bond:
                    # add bonds to the dictionary
                    bonds.setdefault(atom1, []).append(bond)
                    bonds.setdefault(atom2, []).append(bond)

        # self._check_aromaticity(bonds)
        return bonds

    def _update_bonds(self, atom: Atom, action: Literal['add', 'remove']):
        """Update the bonds between atoms after adding or removing atom
        from the molecule.

        Args:
            atom (Atom): Atom to update bonds for.
            action (Literal['add', 'remove']): Action to perform.
        """
        if action == 'add':
            for other_atom in self.atoms:
                if other_atom == atom:
                    continue

                bond = self._validate_bonding(atom, other_atom)
                if bond:
                    self._bonds.setdefault(atom, []).append(bond)
                    self._bonds.setdefault(other_atom, []).append(bond)

        elif action == 'remove':
            if atom in self._bonds:
                for bond in self._bonds[atom]:
                    first_atom, second_atom = bond.atoms
                    other_atom = first_atom if first_atom != atom else second_atom
                    self._bonds[other_atom].remove(bond)
                del self._bonds[atom]

    def add_atom(self, atom: Atom):
        """Add an atom to the molecule.

        Args:
            atom (Atom): Atom to be added to the molecule.
        """
        self.atoms.append(atom)
        self._update_bonds(atom, 'add')

    def remove_atom(self, atom: Atom):
        """Remove an atom from the molecule.

        Args:
            atom (Atom): Atom to be removed from the molecule.
        """
        self.atoms.remove(atom)
        self._update_bonds(atom, 'remove')

    def add_hydrogens(self):
        """Add explicit hydrogens to the molecule"""
        valid_atoms = {'C', 'N', 'O', 'S'}
        distanceHydrogen = Atom(symbol='H').covalent_radius
        for atom in self.atoms:
            if atom.symbol in valid_atoms:
                delocalized_electrons = max(0, sum(bond.aromatic for bond in self.bonds[atom]) - 1)
                n_implicit_Hs = atom.number_of_possible_bonds() \
                                - delocalized_electrons \
                                - sum(
                    bond.order
                    for bond in self.bonds[atom]
                    if not bond.aromatic
                )
                if n_implicit_Hs > 0:
                    # add explicit H atoms such that they lay on a sphere around the atom A
                    # with the radius equal to the sum of covalent radius of the atom A
                    # and hydrogen, and the distance between each hydrogen and other atoms
                    # bonded to the atom A is maximized.
                    constraints = self.atoms

                    pass  # TODO

    @staticmethod
    def _parse_xyz_to_atoms(xyz_string: str) -> List[Atom]:
        """Parse the xyz format (without header) and create atoms from it.

        Args:
            xyz_string (str): String containing the molecule in .xyz format.

        Returns:
            List[Atom]: List of atoms in the molecule.
        """
        xyz_string = xyz_string.strip()
        lines = xyz_string.split('\n')
        atoms = []
        for i, line in enumerate(lines):
            line = line.strip()
            symbol, x, y, z = re.split(r'\s+', line)
            atom = Atom(symbol, name=i + 1, x=float(x), y=float(y), z=float(z))
            atoms.append(atom)

        return atoms

    @classmethod
    def from_xyz(cls, file_path: str) -> Molecule:
        """Create molecule from .xyz file

        Args:
            file_path (str): Path to the .xyz file
        """

        with open(file_path, 'r') as file:
            lines = file.readlines()

        # second line is the (optimal) name of the molecule in xyz format
        name = lines[1].strip()

        xyz_string = ''.join(lines[2:])
        atoms = cls._parse_xyz_to_atoms(xyz_string)

        return cls(name, atoms)

    @classmethod
    def from_mol(cls, file_path: str) -> Molecule:
        """Create molecule from .mol file

        Args:
            file_path (str): Path to the .mol file
        """
        pass  # TODO

    @classmethod
    def from_cif(cls, file_path: str) -> Molecule:
        """Create molecule from .cif file

        Args:
            file_path (str): Path to the .cif file
        """
        pass  # TODO

    @classmethod
    def from_smiles(cls, smiles_string: str) -> Molecule:
        """Create molecule from SMILES string representation

        Args:
            smiles_string (str): SMILES string representation of the molecule
        """
        pass  # TODO

    def to_xyz(self) -> str:
        """Return molecule representation in .xyz format

        Args:
            header (bool, optional): Whether to include the header. Defaults to True.
        Returns:
            str: String representing molecule in .xyz format
        """
        xyz = f'{len(self.atoms)}\n{self.name}\n'  # header
        for atom in self.atoms:
            xyz += f'{atom.symbol} {atom.x} {atom.y} {atom.z}\n'
        return xyz

    def save_xyz(self, file_path: str):
        """Save molecule to .xyz file

        Args:
            file_path (str): Path to the .xyz file
        """
        with open(file_path, 'w') as file:
            file.write(self.to_xyz())

    def to_mol(self) -> str:
        """Return molecule representation in .mol format

        Returns:
            str: String representing molecule in .mol format
        """
        pass  # TODO

    def save_mol(self, file_path: str):
        """Save molecule to .mol file

        Args:
            file_path (str): Path to the .mol file
        """
        with open(file_path, 'w') as file:
            file.write(self.to_mol())

    def to_psi4(self) -> psi4.Molecule:
        """Create psi4 molecule object.

        Returns:
            psi4.Molecule: psi4 molecule object
        """
        psi_molecule = psi4.geometry(self.to_xyz())
        return psi_molecule

    def optimize(
            self,
            method: str = 'scf/cc-pvdz',
            reference: str = 'rhf',
            **kwargs
    ) -> Molecule:
        """Optimize the molecule geometry using psi4.

        Args:
            method (str, optional): Method/basis set to use for geometry
            optimization. Defaults to 'scf/cc-pvdz'.
            reference (str, optional): Reference wavefunction. Defaults to 'rhf'.
            **kwargs: Additional keyword arguments to pass to psi4.set_options.

        Returns:
            Molecule: Optimized molecule
        """
        psi_molecule = self.to_psi4()
        psi4.set_options(
            {
                'reference': reference,
                **kwargs
            })
        psi4.optimize(method, molecule=psi_molecule)
        xyz_string = psi_molecule.save_string_xyz()
        xyz_string = xyz_string.strip().split('\n', 1)[1]  # psi4 returns some header in first line
        atoms = self._parse_xyz_to_atoms(xyz_string)

        return Molecule(self.name, atoms)
