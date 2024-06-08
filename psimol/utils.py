import re
import re
import os 
import sys
import logging 
import argparse
import numpy as np
from ruamel.yaml import YAML
from scipy.optimize import minimize
from typing import Any, Dict, List, Literal, Tuple, Union


class CustomParser(argparse.ArgumentParser):
    """Custom parser class."""
    
    def error(self, message):
        """Print help with error message."""
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)


class CustomLoggingFormatter(logging.Formatter):
    """Custom logging formatter."""

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    blue = "\x1b[36;20m"
    reset = "\x1b[0m"
    format = "[%(asctime)s] [%(levelname)s] %(message)s"

    FORMATS = {
        logging.DEBUG: blue + format + reset,
        logging.INFO: grey + format + reset,
        logging.WARNING: yellow + format + reset,
        logging.ERROR: red + format + reset,
        logging.CRITICAL: bold_red + format + reset
    }
    
    def format(self, record):
        """Format the log record."""
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)
    

def get_package_directory() -> str:
    """Helper function to get the directory of the package.

    Returns:
        str: Path to the package directory.
    """
    return os.path.dirname(os.path.abspath(__file__))


def setup_logging(loglevel: int) -> None:
    """Setup logging for the package.
    
    Args:
        loglevel (int): Log level to set.
    """
    custom_logging = logging.StreamHandler()
    custom_logging.setLevel(loglevel)
    custom_logging.setFormatter(CustomLoggingFormatter())

    logging.basicConfig(
        level=loglevel,
        handlers=[custom_logging],
        force=True    
    )

    # Set logging for psi4 to CRITICAL (it is overdramatic)
    logging.getLogger('psi4').setLevel(logging.CRITICAL)

def get_all_atoms_config() -> Dict[str, Dict[str, Any]]:
    """Get the configuration of all atoms.

    Returns:
        Dict[str, Dict[str, Any]]: Configuration of all atoms.
    """
    package_path = get_package_directory()
    atom_config_path = os.path.join(package_path, 'configs', 'atom_properties.yml')

    if not os.path.exists(atom_config_path):
        logging.fatal(f'Atom configuration file not found at {atom_config_path}.')
        raise FileNotFoundError(f'Atom configuration file not found at {atom_config_path}.')

    yaml = YAML(typ='safe')
    with open(atom_config_path, 'r') as file:
        atom_config = yaml.load(file)
    
    return atom_config

def get_atom_config(symbol: str) -> Dict[str, Any]:
    """Get the configuration of an atom.

    Args:
        symbol (str): Atom symbol.

    Returns:
        Dict[str, Any]: Configuration of an atom.
    """
    atom_config = get_all_atoms_config()

    if symbol not in atom_config:
        logging.error(f'Atom configuration not found for provided symbol {symbol}.')
        raise ValueError(f'Atom configuration not found for provided symbol {symbol}.')
    
    return atom_config[symbol]

def euclidean_distance(coords1: np.ndarray, coords2: np.ndarray) -> float:
    """Calculate the Euclidean distance between two vectors.

    Args:
        coords1 (np.ndarray): First vector coordinates.
        coords2 (np.ndarray): Second vector coordinates.

    Returns:
        float: Euclidean distance between vectors.
    """
    return np.linalg.norm(coords1 - coords2)

def check_smiles_validity(smiles: str) -> bool:
    """Check the validity of a SMILES string.

    Args:
        smiles (str): SMILES string to check.

    Returns:
        bool: True if the SMILES string is valid, False otherwise.
    """
    if not smiles:
        logging.error('Empty SMILES string provided.')
        return False

    if '*' in smiles:
        logging.error('Wildcard character "*" found in SMILES string, which is unsupported.')
        return False
    
    if '.' in smiles:
        logging.error('Multiple molecules (disconnected structures) in SMILES are unsupported.')
        return False
    
    if '$' in smiles:
        logging.error('Quadruple bonds in SMILES are unsupported.')
        return False
    
    if '\\' in smiles or '/' in smiles or '@' in smiles:
        logging.warning('Chirality information in SMILES is unsupported, and will be discarded in geometry computation.')
        # Don't return False here, as it is a warning

    if 'H' in smiles:
        logging.warning('Explicit hydrogens in SMILES are unsupported, and will be discarded during parsing.')
        # Don't return False here, as it is a warning
    
    if re.search(r'[^HBrClNOSPFI\d\W]', smiles, re.IGNORECASE):
        logging.error('Only typically organic (H, C, B, N, O, S, P, F, Br, Cl, and I) atoms are supported in SMILES parsing.')
        return False
    
    return True

def normalize_smiles(smiles: str) -> str:
    """Normalize the SMILES string.

    Args:
        smiles (str): SMILES string to normalize.

    Returns:
        str: Normalized SMILES string.
    """
    # Remove whitespace, ':', '@', '\' and '/' characters
    smiles = re.sub(r'[\s:@\\/]', '', smiles)

    # Remove explicit '-' between non-aromatic atoms
    smiles = re.sub(r'(Br?|Cl?|N|O|S|P|F|I)(-)(Br?|Cl?|N|O|S|P|F|I)', r'\1\3', smiles)

    # Remove leading zeros from numbers while keeping the last digit
    smiles = re.sub(r'0+(?=\d)', '', smiles)

    # Remove explicit hydrogens
    smiles = re.sub(r'H\d*', '', smiles)

    # Change charge representation from [+-]{2,} to [+-]n, like ++ -> +2
    def _replacer(match):
        sign = match.group(0)[0]  # The sign is either '+' or '-'
        count = len(match.group(0))  # The length of the matched group
        return f"{sign}{count}"

    smiles = re.sub(r"(\+{1,}|-{1,})", _replacer, smiles)

    return smiles

def get_optimal_coords(
        n: int, 
        central_point: np.ndarray,
        radius: Union[float, List[float]],
        other_points: np.ndarray
    ) -> np.ndarray:
    """For n points find the optimal coordinates on the sphere of radius r
    centered at central_point, such that the distance to other_points and the pairs 
    of query points is maximized.

    Args:
        initial_points (np.ndarray): Array of points coordinates, of shape (n, 3).
        central_point (np.ndarray): Coordinates of the central point (shape (3)).
        radius (Union[float, List[float]]): Radius of the sphere centered in central_point.
        Can be a single float or a list of floats, one for each point.
        other_points (np.ndarray): Array with coordinates to the other points, of shape (m, 3).

    Returns:
        np.ndarray: Optimal coordinates of the points on the sphere (shape (n, 3)).
    """

    if not isinstance(radius, list):
        radius = [radius] * n
    radius = np.array(radius)

    # Initial guess: n point laying in the central-symmerical reflection of the other_points mass center
    mass_center = np.mean(other_points, axis=0)
    direction = central_point - mass_center
    direction /= np.linalg.norm(direction) if np.linalg.norm(direction) > 0 else 1
    initial_guess = mass_center + radius.reshape(-1, 1) * direction
    initial_guess = initial_guess.reshape(-1)
    initial_guess += np.random.randn(n * 3) * 0.001  # Small random perturbation

    # Objective function: maximize the minimum distance to both other points and among query points
    def objective(p):
        p = p.reshape((n, 3))
        min_distances = []
        
        # Distances to other points
        for i in range(n):
            min_distances.append(np.min(np.linalg.norm(other_points - p[i], axis=1)))
        min_distance_to_other_points = np.min(min_distances)
        # Distances among query points
        if n > 1:
            pairwise_distances = []
            for i in range(n):
                for j in range(i + 1, n):
                    pairwise_distances.append(np.linalg.norm(p[i] - p[j]))
            min_pairwise_distance = np.min(pairwise_distances) / n
        else:
            min_pairwise_distance = 0.0
        # We want to maximize the minimum distances, so we return the negative
        return -(min_distance_to_other_points + min_pairwise_distance)

    # Constraint function: all points must lie on the sphere
    def sphere_constraint(p):
        p = p.reshape((n, 3))
        return [np.linalg.norm(p[i] - central_point) - radius[i] for i in range(n)]

    # Define constraints as a dictionary
    constraints = [
        {
        'type': 'eq', 
        'fun': lambda p, i=i: sphere_constraint(p)[i]
        } 
        for i in range(n)
    ]

    # Run the optimization
    result = minimize(
        objective, 
        initial_guess, 
        constraints=constraints, 
        options={'disp': False}
    )

    # Extract the optimal points
    optimal_points = result.x.reshape((n, 3))
    return optimal_points

def generate_ring_points_with_distances(
        coords: List[Union[None, np.ndarray]], 
        distances: List[float]
    ) -> np.ndarray:
    """Generate coordinates for points on a 3D planar ring based on given distances.

    Args:
        coords (List[Union[None, np.ndarray]]): List of 3D coordinates, where known points 
        are np.ndarray and unknown points are None.
        distances (List[float]): Distances between consecutive points on the ring.

    Returns:
        np.ndarray: Coordinates for all points, calculated if they were originally None.
    """
    N = len(coords)
    
    # Calculate the angles between each point
    circumference = sum(distances)
    angles = np.cumsum([2 * np.pi * dist / circumference for dist in distances])

    # Initialize arrays to store coordinates
    x_coords = np.zeros(N)
    y_coords = np.zeros(N)
    z_coords = np.zeros(N)

    move_index = 0
    flip = False
    known_coords = [coord for coord in coords if coord is not None]
    if not known_coords:
        known_coords = np.zeros((1, 3))
    elif len(known_coords) == 1:
        move_index = next(i for i, coord in enumerate(coords) if coord is not None)

    else:
        origin_known_coords = known_coords = np.array(known_coords)
        sorted_indices = known_coords[:, 0].argsort()

        for col in range(known_coords.shape[1]-1, -1, -1):
            known_coords = known_coords[known_coords[:, col].argsort(kind='mergesort')]

        # check if the path is clockwise or counterclockwise
        if sorted_indices[0] != 0:
            flip = True

        for i, coord in enumerate(coords):
            if np.all(coord == origin_known_coords[0]):
                move_index = i
                break
        
        if flip:
            move_index = len(coords) - move_index - len(known_coords)
        
        # Reinitialize the angles based on the known points
        x_diff = known_coords[1][0] - known_coords[0][0]
        y_diff = known_coords[1][1] - known_coords[0][1]
        angle = np.arctan2(x_diff, y_diff)
        angles = np.cumsum([angle] + [2 * np.pi * dist / circumference for dist in distances[1:]])

    # Coordinates of the first points
    for i, row in enumerate(known_coords):
        x_coords[i] = row[0]
        y_coords[i] = row[1]
        z_coords[i] = row[2]
    
    z0 = z_coords[0]
    # Calculate the coordinates of each point based on the angles and distances
    for i in range(len(known_coords), N):
        x_coords[i] = x_coords[i-1] + distances[i-1] * np.sin(angles[i-1])
        y_coords[i] = y_coords[i-1] + distances[i-1] * np.cos(angles[i-1])
        z_coords[i] = z0  # z-coordinates remain constant
    
    points = np.vstack((x_coords, y_coords, z_coords)).T
    
    # Move the points to the original position
    points = np.roll(points, move_index, axis=0)
    if flip:
        points = np.flip(points, axis=0)

    return points

def optimize_rotation(
        init_coords: np.ndarray, 
        frozen_indices: List[int], 
        other_points: np.ndarray
    ) -> np.ndarray:
    """Optimize the rotation of the non-frozen points in init_coords array, such that the distance
    to other_points is maximized.

    Args:
        init_coords (np.ndarray): Initial coordinates of all points.
        frozen_indices (List[int]): Indices of the frozen points.
        other_points (np.ndarray): Other points to which the distance is maximized.

    Returns:
        np.ndarray: Optimized coordinates of all points.
    """

    frozen_points = init_coords[frozen_indices]
    frozen_mean = np.mean(frozen_points, axis=0)

    # Symmetry point calculation
    non_frozen_indices = [i for i in range(len(init_coords)) if i not in frozen_indices]
    non_frozen_coords = np.array([init_coords[i] for i in non_frozen_indices])

    def rotate_points(coords, angle, center):
        rotation_matrix = np.array([
            [np.cos(angle), -np.sin(angle)],
            [np.sin(angle), np.cos(angle)]
        ])
        centered_coords = coords[:, :2] - center[:2]
        rotated_coords = centered_coords @ rotation_matrix.T
        return rotated_coords + center[:2]
    
    def objective_function(angle):
        rotated_coords = rotate_points(non_frozen_coords, angle, frozen_mean)
        distances = [np.linalg.norm(rotated_coords - other_point[:2], axis=1) for other_point in other_points]
        min_distances = np.min(distances, axis=1)
        return -np.sum(min_distances)  # We want to maximize the distance, so minimize the negative distance

    if len(frozen_points) == 1:
        initial_guess = 0
        result = minimize(objective_function, initial_guess, method='Nelder-Mead')
        optimal_angle = result.x[0]

        # Apply the optimal rotation to non-frozen points
        rotated_coords = rotate_points(non_frozen_coords, optimal_angle, frozen_mean)

        # Construct the final coordinates by combining frozen and rotated non-frozen points
        final_coords = np.array(init_coords)
        for i, idx in enumerate(non_frozen_indices):
            final_coords[idx, :2] = rotated_coords[i]

        return final_coords
    
    elif len(frozen_points) == 2:
        # Check only two points - current position and the reflection
        # with respect to the line passing through the frozen points
        current_distances = [
            np.linalg.norm(non_frozen_coords[:, :2] - other_point[:2], axis=1) 
            for other_point in other_points
        ]
        current_min_distances = np.min(current_distances, axis=1)

        reflected_coords = []
        (x1, y1, _), (x2, y2, _) = frozen_points
        A = y2 - y1
        B = x1 - x2
        C = x2 * y1 - x1 * y2

        # reflection formulas
        d = (A * A + B * B)
        if d == 0:
            logging.warning('Cannot optimize the provided ring coordinates.')
            return init_coords
        for i in range(len(init_coords)):
            x1, y1, z1 = init_coords[i]
            x2 = (B * (B * x1 - A * y1) - A * C) / d
            y2 = (A * (-B * x1 + A * y1) - B * C) / d

            x2 = 2 * x2 - x1
            y2 = 2 * y2 - y1    

            reflected_coords.append([x2, y2, z1])

        reflected_coords = np.array(reflected_coords)
        reflected_distances = [
            np.linalg.norm(reflected_coords[:, :2] - other_point[:2], axis=1) 
            for other_point in other_points
        ]
        reflected_min_distances = np.min(reflected_distances, axis=1)

        if np.sum(reflected_min_distances) > np.sum(current_min_distances):
            return reflected_coords
    return init_coords
