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


def convert_file(input_format, output_format, input_file, output_file):
    """Convert molecule file from one format to another."""
    molecule = None

    # Load molecule from input file
    if input_format == 'mol':
        molecule = Molecule.from_mol(input_file)
    elif input_format == 'xyz':
        molecule = Molecule.from_xyz(input_file)
    elif input_format == 'cif':
        molecule = Molecule.from_cif(input_file)
    elif input_format == 'smiles':
        molecule = Molecule.from_smiles(input_file)
    else:
        raise ValueError(f"Unsupported input format: {input_format}")

    # Save molecule to output file
    if output_format == 'mol':
        molecule.save_mol(output_file)
    elif output_format == 'xyz':
        molecule.save_xyz(output_file)
    else:
        raise ValueError(f"Unsupported output format: {output_format}")


def main():
    args = parse_args()
    if args.command == 'convert':
        convert_file(args.input_format, args.output_format, args.input, args.output)


if __name__ == '__main__':
    main()
