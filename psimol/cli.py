import logging

from .main import Molecule
from .utils import CustomParser, setup_logging


def parse_args():
    """Parse command line arguments."""

    parser = CustomParser(description='psimol command line interface')
    parser.add_argument(
        '-v',
        '--verbose',
        action='store_true',
        help='Enable verbose logging.'
    )
    parser.add_argument(
        '-q',
        '--quiet',
        action='store_true',
        help='Disable logging except errors.'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to run.')

    convert_parser = subparsers.add_parser('convert', help='Convert molecule file formats')
    convert_parser.add_argument(
        '--input-format',
        choices=['xyz', 'cif', 'smiles', 'mol'],
        required=True,
        help='Input file format.'
    )
    convert_parser.add_argument(
        '--output-format',
        choices=['xyz', 'mol'],
        required=True,
        help='Output file format.'
    )
    convert_parser.add_argument(
        '-i', '--input',
        type=str,
        required=True,
        help='Path to the input file.'
    )
    convert_parser.add_argument(
        '-o', '--output',
        type=str,
        required=True,
        help='Path to the output file.'
    )

    convert_parser.add_argument(
        '--add-hydrogens',
        action='store_true',
        help='Add hydrogens to the molecule.'
    )

    args = parser.parse_args()
    if not args.command:
        parser.print_help()
        parser.exit(1)

    if args.verbose:
        setup_logging(logging.DEBUG)
    elif args.quiet:
        setup_logging(logging.ERROR)
    else:
        setup_logging(logging.INFO)

    return args

def load_molecule(input_format, input_file):
    if input_format == 'mol':
        molecule = Molecule.from_mol(input_file)
    elif input_format == 'xyz':
        molecule = Molecule.from_xyz(input_file)
    elif input_format == 'cif':
        molecule = Molecule.from_cif(input_file)
    elif input_format == 'smiles':
        with open(input_file, 'r') as f:
            input_file = f.readlines()
        molecule = []
        for line in input_file:
            molecule.append(Molecule.from_smiles(line.strip()))
    else:
        raise ValueError(f"Unsupported input format: {input_format}")
    
    return molecule

def save_molecule(molecule, output_format, output_file):
    if isinstance(molecule, list):
        for i, mol in enumerate(molecule):
            if output_format == 'mol':
                mol.save_mol(output_file + f'_{i}.mol')
            elif output_format == 'xyz':
                mol.save_xyz(output_file + f'_{i}.xyz')
            else:
                raise ValueError(f"Unsupported output format: {output_format}")
    else:
        if output_format == 'mol':
            molecule.save_mol(output_file)
        elif output_format == 'xyz':
            molecule.save_xyz(output_file)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")

def convert_file(input_format, output_format, input_file, output_file, add_hydrogens=False):
    """Convert molecule file from one format to another."""
    molecule = load_molecule(input_format, input_file)

    if add_hydrogens and input_format != 'smiles':
        molecule.add_hydrogens()

    save_molecule(molecule, output_format, output_file)

def main():
    args = parse_args()
    if args.command == 'convert':
        convert_file(args.input_format, args.output_format, args.input, args.output)


if __name__ == '__main__':
    main()
