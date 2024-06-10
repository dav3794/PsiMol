from __future__ import annotations
import re
import psi4
import py3Dmol
import logging
import numpy as np

from typing import Dict, FrozenSet, List, Literal, Tuple, Union, Set
from collections import OrderedDict

from .utils import (
    get_atom_config, 
    setup_logging, 
    euclidean_distance, 
    check_smiles_validity,
    normalize_smiles,
    get_optimal_coords,
    generate_ring_points_with_distances,
    optimize_rotation
    )

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
            raise ValueError('Cannot create a bond of an atom to itself.')
        
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
        
        # Attributes from psi4 calculations
        self.energy = None
        self.wfn = None
        self.frequencies = None

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
    
    @property
    def coord_matrix(self) -> np.ndarray:
        """Return the coordinates of all atoms in the molecule
        as a matrix.

        Returns:
            np.ndarray: Matrix with coordinates of all atoms
        """
        return np.stack([atom.xyz for atom in self.atoms])
        
    def get_atoms_within_distance(self, point: np.ndarray, distance: float) -> List[Atom]:
        """Get all atoms within a given distance from a point.

        Args:
            point (np.ndarray): Coordinates of the point.
            distance (float): Distance (radius) to the point.

        Returns:
            List[Atom]: List of atoms within the distance from the point.
        """
        atom_distances = np.linalg.norm(self.coord_matrix - point, axis=1)
        atoms_within_distance = [
            atom
            for atom, atom_distance in zip(self.atoms, atom_distances)
            if atom_distance <= distance
        ]
        return atoms_within_distance
    
    def get_bonded_atoms(self, atom: Atom) -> List[Atom]:
        """Get all atoms bonded to the given atom.

        Args:
            atom (Atom): Atom to get bonded atoms for.

        Returns:
            List[Atom]: List of atoms bonded to the given atom.
        """
        bonded_atoms = [
            other_atom
            for bond in self.bonds[atom]
            for other_atom in bond.atoms
            if other_atom != atom
        ]
        return bonded_atoms

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

    def _find_cycles(
            self, 
            bonds: Dict[Atom, List[Bond]], 
            aromatic_atoms: Set[str] = {'C', 'N', 'O', 'S'}
        ) -> List[List[Atom]]:
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
        p2 = cycle[2].xyz
        normal = np.cross(p1 - p0, p2 - p0)
        for i in range(3, len(cycle)):
            if not np.isclose(np.dot(normal, cycle[i].xyz - p0), 0.0, atol=0.2):
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
        all_cycles = set(frozenset(cycle) for cycle in all_cycles)
        all_cycles = [list(cycle) for cycle in all_cycles]

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

        self._check_aromaticity(bonds)
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
        distance_hydrogen = Atom(symbol='H').covalent_radius[0]
        for atom in self.atoms:
            if atom.symbol in valid_atoms:
                delocalized_electrons = max(0, sum(bond.aromatic for bond in self.bonds[atom]) - 1)
                n_implicit_Hs = atom.number_of_possible_bonds() \
                                - delocalized_electrons \
                                - sum(
                    bond.order
                    if not bond.aromatic
                    else 1
                    for bond in self.bonds[atom] 
                )
                if n_implicit_Hs > 0:
                    # add explicit H atoms such that they lay on a sphere around the atom A
                    # with the radius equal to the sum of covalent radius of the atom A
                    # and hydrogen, and the distance between each hydrogen and other atoms
                    # bonded to the atom A is maximized.
                    logging.debug(f'Adding {n_implicit_Hs} hydrogens to {atom}')
                    bond_length = atom.covalent_radius[0] + distance_hydrogen
                    bonded_atoms = set(self.get_bonded_atoms(atom))

                    constraints_bonded = [
                        atom.xyz
                        for atom in bonded_atoms
                    ]
                    constraints_nonbonded = [
                        nonb_atom.xyz
                        for nonb_atom in self.get_atoms_within_distance(atom.xyz, 2 * bond_length)
                        if nonb_atom not in bonded_atoms and nonb_atom != atom
                    ]

                    other_points = np.stack(constraints_bonded + constraints_nonbonded)

                    optimal_coords = get_optimal_coords(
                        n=n_implicit_Hs,
                        central_point=atom.xyz,
                        radius=bond_length,
                        other_points=other_points
                    )

                    for coords in optimal_coords:
                        hydrogen = Atom(symbol='H', x=coords[0], y=coords[1], z=coords[2])
                        bond = Bond(atom, hydrogen, order=1)
                        self._atoms.append(hydrogen)
                        self._bonds.setdefault(atom, []).append(bond)
                        self._bonds.setdefault(hydrogen, []).append(bond)

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
        
        # second line is the (optional) name of the molecule in xyz format
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
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # first line of file may or may not contain molecule name
        name = lines[0].strip()
        # fourth line contains atom and bond count
        counts = lines[3].strip()

        # fields in .mol files are fixed-width, rather than delineated by whitespace
        atom_cnt = int(counts[0:3])
        bond_cnt = int(counts[3:6])

        atom_block = lines[4          : 4+atom_cnt]
        bond_block = lines[4+atom_cnt : 4+atom_cnt+bond_cnt]

        atoms = []
        # used to convert between .mol file symbolic representation of atomic charges to actual values
        actual_charges = { '  0': 0, '  1': 3, '  2': 2, '  3': 1, '  4': 0, '  5': -1, '  6': -2, '  7': -3 }
        for i, line in enumerate(atom_block):
            x = line[ 0:10]
            y = line[10:20]
            z = line[20:30]
            symbol = line[31:34].rstrip()
            charge = actual_charges[line[36:39]]
            atom = Atom(symbol, name=i+1, x=float(x), y=float(y), z=float(z), charge=charge)
            atoms.append(atom)

        bonds: Dict[Atom, List[Bond]] = { atom:[] for atom in atoms }
        for line in bond_block:

            atom1_idx = int(line[0:3])
            atom2_idx = int(line[3:6])
            bond_type = int(line[6:9])
            if bond_type < 4:
                bond_order = bond_type
                bond_aromaticity = False
            elif bond_type == 4:
                bond_order = 1
                bond_aromaticity = True
            else:
                bond_order = 1
                bond_aromaticity = False

            # atoms in the bond block are 1-indexed
            bond = Bond(atoms[atom1_idx-1], atoms[atom2_idx-1], bond_order, bond_aromaticity)
            bonds[atoms[atom1_idx-1]].append(bond)
            bonds[atoms[atom2_idx-1]].append(bond)

        return cls(name, atoms, bonds)

    @classmethod
    def from_cif(cls, file_path: str) -> Molecule:
        """Create molecule from .cif file

        Args:
            file_path (str): Path to the .cif file
        """        
        with open(file_path, 'r') as file:
            lines = file.readlines()

        # the parser will advance to the 'atom_site' loop of the .cif file
        # (in STAR files (.cif included), tables are called "loops"),
        # assemble the loop's contents, then parse them

        # used to test if a line is from the header of the 'atom_site' loop
        def _is_atom_site_header(s: str) -> bool:
            return bool(re.match(r'_atom_site[_.]', s, re.IGNORECASE))
        # parser state variables
        in_atom_site = in_atom_site_body = False

        # will hold the columns of the 'atom_site' loop, along with their order
        atom_site_header: dict[str:int] = {}
        ncol = 0
        # columns required to be present by the parser
        required_columns = {'type_symbol', 'cartn_x', 'cartn_y', 'cartn_z'}
        # will hold the body of the 'atom_site' loop
        atom_site_body: list[str] = ""

        for line in lines:
            line = line.lstrip()

            # phase 1: advance parser to 'atom_site' loop
            # (in STAR files (CIF included), tables are called "loops")
            if not in_atom_site:
                in_atom_site = _is_atom_site_header(line)

            # phase 2: assemble the 'atom_site' loop
            if in_atom_site:
                
                # first, retrieve the columns of the loop
                if not in_atom_site_body:
                    if _is_atom_site_header(line):
                        # add column name to the list, without the loop name prefix
                        column_name = line[11:].lower().rstrip()
                        if columns_name in required_columns:
                            atom_site_header[column_name] = ncol
                        ncol += 1
                    else:
                        in_atom_site_body = True
                        # check that the required columns are present
                        if len(atom_site_header < 4):
                            logging.error(f'.cif file\'s atom_site section lacks the required columns')
                            return None
                
                # assemble the body of the loop
                if in_atom_site_body:
                    # a loop may be terminated by:
                    # - a line containing a single '#' and otherwise empty
                    # - a new non-looped data item
                    # - an empty line
                    # - start of a new loop
                    if line[0] == '#' or line[0] == '_' or line == '' or line[:5].lower() == 'loop_':
                        break
                    atom_site_body += line

        # phase 3: parse loop
        # .cif file specification states a length limit for lines.
        # To satisfy this limit, overlong records in loops may be broken into multiple lines.
        # This caveat essentially makes the newline no different from any other whitespace,
        # which necessitates treating the body of the loop in a flat manner.

        # guaranteed to mangle any quoted items containing whitespace
        atom_site_body = atom_site_body.split()
        atoms = []
        symbol_idx = atom_site_header['type_symbol']
        x_idx = atom_site_header['cartn_x']
        y_idx = atom_site_header['cartn_y']
        z_idx = atom_site_header['cartn_z']
        nrow = len(atom_site_body) // ncol
        # check that the number of items is as would be expected based on the number of rows and columns
        if nrow*ncol != len(atom_site_body):
            # if it isn't, a possible cause is that a quoted item got mangled
            logging.error('Failed to parse .cif file, possibly incompatible')
            # since this would mess up all the remaining items, the parser gives up
            return None

        for i in range(nrow):

            offset = i*ncol

            # extract atom's items from `body'
            symbol: str = atom_site_body[offset+symbol_idx]
            x: str = atom_site_body[offset+x_idx]
            y: str = atom_site_body[offset+y_idx]
            z: str = atom_site_body[offset+z_idx]

            # '.' and '?' signify unapplicable and missing values
            # the parser can't cope with missing coord or element info and gives up
            if any(( item == '?' or item == '.' for item in (symbol, x, y, z) )):
                logging.error('.cif file is missing an atom\'s element or coordinate')
                return None

            # numeric values in .cif files may be appended with an uncertainty in parenthesis
            # the parser ignores this
            x = re.match(r'^[^(]+', x)
            y = re.match(r'^[^(]+', y)
            z = re.match(r'^[^(]+', z)
            
            atom = Atom(symbol, name=i+1, x=float(x), y=float(y), z=float(z))
            atoms.append(atom)

        # get conjectural name
        name = os.path.basename(file_path)

        return cls(name, atoms)

    @classmethod
    def from_smiles(cls, smiles_string: str) -> Molecule:
        """Create molecule from SMILES string representation

        Args:
            smiles_string (str): SMILES string representation of the molecule
        """
        if not check_smiles_validity(smiles_string):
            raise ValueError('Invalid SMILES string provided.')
        
        smiles_string = normalize_smiles(smiles_string)
        
        # simplified tokenizing pattern from Molecular Transformer 
        # (https://github.com/pschwllr/MolecularTransformer)
        pattern =  r"(\[[^\]]+]|Br?|Cl?|N|O|S|P|F|I|b|c|n|o|s|p|\(|\)|=|#|-|\+|\%[0-9]{2}|[0-9])"
        regex = re.compile(pattern)
        tokens = [token for token in regex.findall(smiles_string)]

        isotopic_mass_pattern = re.compile(r'^\[([0-9]+)')
        charge_pattern = re.compile(r'([+-][0-9]+)]$')
        bracketed_token_symbol = re.compile(r'[A-GI-Za-z][a-z]?') # exclude H

        # Create atoms from tokens
        atoms = []
        bonds = {}
        bond_order = 1
        last_atom_idx = 0
        current_atom_idx = 0

        branch_stack = []
        ring = {} # dict mapping ring number to atom
        
        add_atom = False
        aromatic_atoms: List[bool] = [] # whether i-th atom is aromatic
        for token in tokens:
            if token[0] == '[': # bracketed tokens like [C], [C+], [C+2], [13C]

                symbol = bracketed_token_symbol.findall(token)[0]
                atom = Atom(symbol.capitalize(), name=current_atom_idx+1)

                isotopic_mass = isotopic_mass_pattern.findall(token)
                if isotopic_mass:
                    atom.mass = int(isotopic_mass[0])
                
                charge = charge_pattern.findall(token)
                if charge:
                    atom.charge = int(charge[0])
    
                add_atom = True

            elif token.isalpha(): # atom symbol
                symbol = token
                atom = Atom(symbol.capitalize(), name=current_atom_idx+1)
                add_atom = True                    

            elif token in {'=', '#'}: # double or triple bond
                bond_order = 2 if token == '=' else 3

            elif token == '(': # branch start
                branch_stack.append(last_atom_idx)
            
            elif token == ')': # branch end
                last_atom_idx = branch_stack.pop()
            
            elif token.isdigit() or token[0] == '%': # ring number
                if token in ring:
                    ring_atom = ring[token]
                    del ring[token]
                    if aromatic_atoms[ring_atom] and aromatic_atoms[last_atom_idx]:
                        aromatic = True
                    else:
                        aromatic = False
                    bond = Bond(
                        atoms[ring_atom], 
                        atoms[last_atom_idx], 
                        order=bond_order,
                        aromatic=aromatic
                    )
                    bonds[atoms[ring_atom]].append(bond)
                    bonds[atoms[last_atom_idx]].append(bond)
                else:
                    ring[token] = last_atom_idx
            
            if add_atom:
                atoms.append(atom)
                aromatic_atoms.append(symbol.islower())
                bonds[atom] = []

                if current_atom_idx:
                    if aromatic_atoms[last_atom_idx] and aromatic_atoms[-1]:
                        aromatic = True
                    else:
                        aromatic = False
                    bond = Bond(
                        atoms[last_atom_idx], 
                        atom, 
                        order=bond_order, 
                        aromatic=aromatic
                    )
                    bonds[atoms[last_atom_idx]].append(bond)
                    bonds[atom].append(bond)
                    last_atom_idx = current_atom_idx
                current_atom_idx += 1
                add_atom = False
                bond_order = 1

        molecule = cls(smiles_string, atoms, bonds)

        # Create planar geometry
        rings = molecule._find_cycles(bonds)
        if rings:
            rings_dict = OrderedDict()
            for ring in rings:
                rings_dict[frozenset(ring)] = ring
            
            unique_rings = []
            for ring1 in rings_dict.keys():
                for ring2 in rings_dict.keys():
                    if ring1 == ring2:
                        continue
                    if ring2.issubset(ring1) or \
                       (len(ring1 & ring2) > 3 and len(ring1) >= 2 * len(ring2)): # for fused rings
                        break
                else:
                    unique_rings.append(rings_dict[ring1])

            rings = unique_rings

        ringed_atoms: Dict[Atom, Set[int]] = {} # dict mapping atoms to the indices of the ring(s) they belong to
        for i, ring in enumerate(rings):
            for atom in ring:
                if atom in ringed_atoms:
                    ringed_atoms[atom].add(i)
                else:
                    ringed_atoms[atom] = {i}       
        
        seen_atoms = set()
        for atom in atoms:            
            atom_bonds = bonds[atom]

            if len(atom_bonds) == 1: # case of non-branched first atom
                bond = atom_bonds[0]
                other_atom = bond.atoms[0] if bond.atoms[0] != atom else bond.atoms[1]
                if other_atom in seen_atoms:
                    continue
                
                bond_length = atom.covalent_radius[bond.order - 1] \
                                + other_atom.covalent_radius[bond.order - 1]
                
                other_atom.x = atom.x + bond_length # simply shift the atom in x direction
                seen_atoms.add(other_atom)

            elif len(atom_bonds) > 1:
                bonded_atoms_to_optimize = []
                other_points = []
                radii = []
                for bond in atom_bonds:
                    other_atom = bond.atoms[0] if bond.atoms[0] != atom else bond.atoms[1]
                    if other_atom in seen_atoms:
                        other_points.append(other_atom.xyz)
                        continue

                    if other_atom in ringed_atoms and atom in ringed_atoms and ringed_atoms[atom] & ringed_atoms[other_atom]:
                        ring_index = (ringed_atoms[atom] & ringed_atoms[other_atom]).pop() # index of intersecting ring
                        ring = rings[ring_index]
                        shift = ring.index(other_atom)
                        ring_coords, ring_distances = [], []

                        # shift the ring so that the current atom is the first one
                        ring_atoms = ring[shift:] + ring[:shift]
                        for idx, ring_atom in enumerate(ring_atoms):
                            other_points.append(ring_atom.xyz)

                            ring_coords.append(
                                ring_atom.xyz 
                                if sum(ring_atom.xyz) != 0 # assume that point zero is an unassigned point
                                else None
                            )

                            # calculate supposed bond length
                            previous_ring_atom = ring_atoms[idx - 1]
                            for ring_bond in bonds[ring_atom]:
                                if previous_ring_atom in ring_bond.atoms:
                                    bond_order = ring_bond.order
                                    break
                            else:
                                raise ValueError('No bond found between ring atoms.')
                            
                            bond_length = previous_ring_atom.covalent_radius[bond_order - 1] \
                                            + ring_atom.covalent_radius[bond_order - 1]
                            ring_distances.append(bond_length)
                        
                    
                        ring_points = generate_ring_points_with_distances(
                            ring_coords, 
                            ring_distances
                        )

                        # optimize the ring geometry
                        frozen_points_idx = [idx for idx, point in enumerate(ring_coords) if point is not None]
                        if frozen_points_idx:
                            seen_points = np.stack([at.xyz for at in seen_atoms])
                            ring_points = optimize_rotation(
                                ring_points, 
                                frozen_points_idx, 
                                seen_points
                            )

                        for ring_atom, ring_atom_coords in zip(ring_atoms, ring_points):
                            if ring_atom not in seen_atoms:
                                ring_atom.x, ring_atom.y, ring_atom.z = ring_atom_coords
                                seen_atoms.add(ring_atom)
                            ringed_atoms[ring_atom].remove(ring_index)

                    else:
                        bonded_atoms_to_optimize.append(other_atom)

                        bond_length = atom.covalent_radius[bond.order - 1] \
                                        + other_atom.covalent_radius[bond.order - 1]
                        radii.append(bond_length)
                        seen_atoms.add(other_atom)
                    
                if not bonded_atoms_to_optimize:
                    continue

                if not other_points:
                    other_points = np.array([atom.xyz])
                else:
                    other_points = np.stack(other_points)

                optimal_coords = get_optimal_coords(
                    n=len(bonded_atoms_to_optimize),
                    central_point=atom.xyz,
                    radius=radii,
                    other_points=other_points
                )
                
                for other_atom, coords in zip(bonded_atoms_to_optimize, optimal_coords):
                    other_atom.x, other_atom.y, other_atom.z = coords
            seen_atoms.add(atom)

        molecule.add_hydrogens()
        return molecule
    
    def visualize(self):
        """Visualize the molecule using py3Dmol."""
        mol = self.to_mol()
        molview = py3Dmol.view(width=400, height=400)
        molview.addModel(mol)
        molview.setStyle({'stick':{}})
        molview.zoomTo()
        molview.show()

    def show_modes(self, mode: int = 0):
        """Show the vibrational mode of the molecule using py3Dmol.

        Args:
            mode (int, optional): Mode to show. Defaults to 0.
        """
        if self.frequencies is None:
            logging.error('No vibrational frequencies found. Make sure to run ' + \
                            'molecule.calculate_frequencies() on optimized molecule.')
            return

        if mode >= len(self.frequencies):
            logging.error('Mode index out of range.')
            return

        modes = self.frequencies[mode]
        xyz = self.to_xyz()
        lines = xyz.split('\n')
        new_lines = [
            lines[0],
            lines[1]
        ]
        for mod, atom_coords in zip(modes, lines[2:]):
            new_lines.append(atom_coords + ' ' + ' '.join(str(x) for x in mod))

        xyz_with_freqs = '\n'.join(new_lines)
        xyzview = py3Dmol.view(width=400,height=400)

        xyzview.addModel(xyz_with_freqs, 'xyz', {'vibrate': {'frames': 10, 'amplitude': 1}})
        xyzview.setStyle({'stick': {}})
        xyzview.animate({'loop': 'backAndForth'})
        xyzview.zoomTo()
        xyzview.show()

    def to_xyz(self) -> str:
        """Return molecule representation in .xyz format

        Args:
            header (bool, optional): Whether to include the header. Defaults to True.
        Returns:
            str: String representing molecule in .xyz format
        """
        xyz = f'{len(self.atoms)}\n{self.name}\n'  # header
        for atom in self.atoms:
            xyz += f'{atom.symbol} {atom.x:.5f} {atom.y:.5f} {atom.z:.5f}\n'
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
        # header
        out = self.name + "\n\n\n"

        unique_bonds = set()
        for bonds in self.bonds.values():
            for bond in bonds:
                if bond not in unique_bonds:
                    unique_bonds.add(bond)

        # construct atom-to-index dictionary, needed for generating the bond block
        atom_indices: dict[Atom:int] = { self._atoms[i]:i+1 for i in range(len(self._atoms)) }

        # counts line
        out += f'{len(self._atoms):3}{len(unique_bonds):3}  0  0  0  0  0  0  0  0999 V2000\n'

        # used to convert between actual atomic charges to .mol file symbolic representations
        charge_indices = { 0: '0', 1: '3', 2: '2', 3: '1', -1: '5', -2: '6', -3: '7'  }
        # atoms
        for atom in self._atoms:
            charge = atom.charge if -3 <= atom.charge <= 3 else 0
            charge = charge_indices[charge]
            out += f'{atom.x:10.4f}{atom.y:10.4f}{atom.z:10.4f} {atom.symbol:3} 0  {charge}  0  0  0  0  0  0  0  0  0\n'

        #bonds
        for bond in unique_bonds:
            atom1_idx, atom2_idx = ( atom_indices[atom] for atom in bond.atoms )
            if bond.aromatic:
                bond_type = 4
            else:
                bond_type = bond.order
            out += f'{atom1_idx:3}{atom2_idx:3}{bond_type:3}  0  0  0  0\n'

        out += "M  END\n"
        return out

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
            method: str = 'b3lyp/6-31g*',
            num_threads: int = 4,
            memory: str = '2GB',
            **kwargs
    ) -> Tuple[Molecule, float, psi4.core.Wavefunction]:
        """Optimize the molecule geometry using psi4.

        Args:
            method (str, optional): Method/basis set to use for geometry
            optimization. Defaults to 'b3lyp/6-31g*'.
            num_threads (int, optional) Number of threads. Defaults to 4.
            memory (str, optional): Memory to allocate for the computation. Defaults to '2GB'.
            **kwargs: Additional keyword arguments to pass to psi4.set_options.

        Returns:
            Tuple[Molecule, float]: Optimized molecule and its energy.
        """
        psi_molecule = self.to_psi4()
        psi4.set_options({**kwargs})
        psi4.set_memory(memory)
        psi4.core.be_quiet()
        psi4.core.set_num_threads(num_threads)
        try:
            energy, wfn = psi4.optimize(method, molecule=psi_molecule, return_wfn=True)
        except Exception as e:
            logging.error(f'Calculation did not converge. Try again with a different method or parameters.\n\n{e}')
            return None
            
        xyz_string = psi_molecule.save_string_xyz()
        xyz_string = xyz_string.strip().split('\n', 1)[1]  # psi4 returns some header in first line
        atoms = self._parse_xyz_to_atoms(xyz_string)
        logging.info(f'Optimized geometry of {self.name} with energy {energy:.5f} Ha.')

        molecule = Molecule(self.name, atoms)
        molecule.energy = energy
        molecule.wfn = wfn

        return molecule, energy, wfn
    
    def calculate_frequencies(
            self,
            method: str = 'b3lyp/6-31g*',
            num_threads: int = 4,
            memory: str = '2GB',
            **kwargs
    ) -> Tuple[float, np.ndarray, psi4.core.Wavefunction]:
        """Calculate vibrational frequencies of the molecule using psi4.

        Args:
            method (str, optional): Method/basis set to use for vibrational
            frequency calculation. Defaults to 'b3lyp/6-31g*'.
            num_threads (int, optional) Number of threads. Defaults to 4.
            memory (str, optional): Memory to allocate for the computation. Defaults to '2GB'.
            **kwargs: Additional keyword arguments to pass to psi4.set_options.

        Returns:
            Tuple[float, np.ndarray]: Energy and modes of vibrational frequencies.
        """
        psi_molecule = self.to_psi4()
        psi4.set_options({**kwargs}, verbose=False)
        psi4.set_memory(memory)
        psi4.core.be_quiet()
        psi4.core.set_num_threads(num_threads, quiet=True)
        try:
            energy, wfn = psi4.frequency(method, molecule=psi_molecule, return_wfn=True)
        except Exception as e:
            logging.error(f'Calculation did not converge. Try again with a different method or parameters.\n\n{e}')
            return None

        logging.info(f'Calculated vibrational frequencies of {self.name} with  energy {energy:.5f} Ha.')

        frequencies = wfn.frequency_analysis['x'].data
        frequencies = frequencies.reshape(-1, len(self.atoms), 3)

        if self.frequencies is None:
            self.frequencies = frequencies
        
        if self.energy is None:
            self.energy = energy
        
        if self.wfn is None:
            self.wfn = wfn

        return energy, frequencies, wfn
    
    def calculate_energy(
            self,
            method: str = 'b3lyp/6-31g*',
            num_threads: int = 4,
            memory: str = '2GB',
            **kwargs
    ) -> float:
        """Calculate single-point energy of the molecule using psi4.

        Args:
            method (str, optional): Method/basis set to use for calculations. Defaults to 'b3lyp/6-31g*'.
            num_threads (int, optional): Number of threads. Defaults to 4.
            memory (str, optional): Amount of memory to reserve. Defaults to '2GB'.

        Returns:
            float: Single point energy of the molecule in Hartree.
        """
        psi_molecule = self.to_psi4()
        psi4.set_options({**kwargs})
        psi4.set_memory(memory)
        psi4.core.be_quiet()
        psi4.core.set_num_threads(num_threads)
        try:
            energy = psi4.energy(method, molecule=psi_molecule)
        except Exception as e:
            logging.error(f'Calculation did not converge. Try again with a different method or parameters.\n\n{e}')
            return None

        logging.info(f'Calculated single-point energy of {self.name} equals {energy:.5f} Ha.')

        if self.energy is None:
            self.energy = energy
        
        return energy
