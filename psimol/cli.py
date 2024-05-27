import logging

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

    ... #TODO
    
    args = parser.parse_args()
    if not args:
        parser.print_help()
        parser.exit(1)

    if args.verbose:
        setup_logging(logging.DEBUG)
    elif args.quiet:
        setup_logging(logging.ERROR)
    else:
        setup_logging(logging.INFO)

    return args


def main():
    args = parse_args()
    

if __name__ == '__main__':
    main()