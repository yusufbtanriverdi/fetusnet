import argparse
from net.parameters.config import parameters_default, parameters_choices, parameters_help, print_help


def parameters_parsing() -> argparse.Namespace:
    """
    Definition of parameters-parsing for each execution mode

    :return: parser of parameters parsing
    """

    # Initialize the main argument parser
    parser = argparse.ArgumentParser(description='Argument Parser')

    # ---------------- #
    # EXECUTION MODES  #
    # ---------------- #
    parser_mode = parser.add_subparsers(title=parameters_help['mode'], dest='mode', metavar='mode')

    # Define subparsers for different execution modes
    parser_train = parser_mode.add_parser('train', help=parameters_help['train'])
    parser_test = parser_mode.add_parser('test', help=parameters_help['test'])
    parser_train_test = parser_mode.add_parser('train_test', help=parameters_help['train_test'])
    parser_explainability = parser_mode.add_parser('explainability', help=parameters_help['explainability'])

    # Define subparsers for utility scripts
    parser_script_prepare = parser_mode.add_parser('script_prepare', help=parameters_help['script_prepare'])
    parser_script_split = parser_mode.add_parser('script_split', help=parameters_help['script_split'])
    parser_script_rotate = parser_mode.add_parser('script_rotate', help=parameters_help['script_rotate'])
    parser_script_generate_targets = parser_mode.add_parser('script_generate_targets', help=parameters_help['script_generate_targets'])

    # Help subparser
    parser_script_help = parser_mode.add_parser('help', help=parameters_help['help'])
    parser_script_help.set_defaults(func=print_help)

    # List of all execution modes
    execution_mode = [
        parser_train,
        parser_test,
        parser_train_test,
        parser_explainability,
        parser_script_split,
        parser_script_prepare,
        parser_script_rotate,
        parser_script_generate_targets,
        parser_script_help
    ]

    # ----------------------- #
    # COMMON PARAMETERS GROUP #
    # ----------------------- #
    for subparser in execution_mode:
        # Initialization parameters
        subparser.add_argument('--root', type=str, default=parameters_default['root'], help=parameters_help['root'])
        subparser.add_argument('--raw_dir', type=str, default=parameters_default['raw_dir'], help=parameters_help['raw_dir'])
        subparser.add_argument('--os', type=str, default=parameters_default['os'], help=parameters_help['os'])
        subparser.add_argument('--wandbpro', type=str, default=parameters_default['wandbpro'], help=parameters_help['wandbpro'])
        # Device parameter
        subparser.add_argument('--device', type=str, default=parameters_default['device'], help=parameters_help['device'])
        # Rotation parameters
        subparser.add_argument('--desired_size', type=int, default=parameters_default['desired_size'], help=parameters_help['desired_size'])
        subparser.add_argument('--desired_spacings', type=int, default=parameters_default['desired_spacings'], help=parameters_help['desired_spacings'])
        subparser.add_argument('--drate', type=int, default=parameters_default['drate'], help=parameters_help['drate'])

        # Dataset parameters
        subparser.add_argument('--dataset', nargs='+', type=str, default=parameters_default['dataset'], help=parameters_help['dataset'])
        subparser.add_argument('--n_split', type=int, default=parameters_default['n_split'], help=parameters_help['n_split'])
        subparser.add_argument('--test_patients', nargs='+', type=int, default=parameters_default['test_patients'], help=parameters_help['test_patients'])
        subparser.add_argument('--test_fold', '-f', type=int, default=parameters_default['test_fold'], help=parameters_help['test_fold'])

        # Target generation parameters
        subparser.add_argument('--generate', type=str, default=parameters_default['generate'], help=parameters_help['generate'])
        subparser.add_argument('--target_idx', nargs='+', type=int, default=parameters_default['target_idx'], help=parameters_help['target_idx'])
        subparser.add_argument('--alpha', type=float, default=parameters_default['alpha'], help=parameters_help['alpha'])
        subparser.add_argument('--eps', type=int, default=parameters_default['eps'], help=parameters_help['eps'])

        # Experiment parameters
        subparser.add_argument('--run_name', type=str, default=parameters_default['run_name'], help=parameters_help['run_name'])
        subparser.add_argument('--use_wandb', action='store_true', default=parameters_default['use_wandb'], help=parameters_help['use_wandb'])

        # Reproducibility
        subparser.add_argument('--seed', type=int, default=parameters_default['seed'], help=parameters_help['seed'])

        # Dataset normalization and transforms
        subparser.add_argument('--rescale', action='store_true', default=parameters_default['rescale'], help=parameters_help['rescale'])
        subparser.add_argument('--lmks', type=str, nargs='+', default=parameters_default['lmks'], help=parameters_help['lmks'])

        # Data loader parameters
        subparser.add_argument('--batch_size_train', '--bs', type=int, default=parameters_default['batch_size_train'], help=parameters_help['batch_size_train'])
        subparser.add_argument('--batch_size_val', type=int, default=parameters_default['batch_size_val'], help=parameters_help['batch_size_val'])
        subparser.add_argument('--batch_size_test', type=int, default=parameters_default['batch_size_test'], help=parameters_help['batch_size_test'])
        subparser.add_argument('--num_workers', type=int, default=parameters_default['num_workers'], help=parameters_help['num_workers'])

        # Network parameters
        subparser.add_argument('--backbone', type=str, default=parameters_default['backbone'], help=parameters_help['backbone'])
        subparser.add_argument('--training_mode', type=str, default=parameters_default['training_mode'], help=parameters_help['training_mode'])

        # Hyperparameters
        subparser.add_argument('--epochs', '--ep', type=int, default=parameters_default['epochs'], help=parameters_help['epochs'])
        subparser.add_argument('--num_fts', type=int, default=parameters_default['num_fts'], help=parameters_help['num_fts'])
        subparser.add_argument('--optimizer', '--optim', type=str, default=parameters_default['optimizer'], choices=parameters_choices['optimizer'], help=parameters_help['optimizer'])
        subparser.add_argument('--learning_rate', '--lr', type=float, default=parameters_default['learning_rate'], help=parameters_help['learning_rate'])
        subparser.add_argument('--lr_momentum', '-m', type=int, default=parameters_default['lr_momentum'], help=parameters_help['lr_momentum'])

        # Loss parameters
        subparser.add_argument('--reduction', type=str, choices=parameters_choices['reduction'], default=parameters_default['reduction'], help=parameters_help['reduction'])
        subparser.add_argument('--loss', '-l', nargs='+', type=str, default=parameters_default['loss'], help=parameters_help['loss'])

        # Output parameters
        subparser.add_argument('--type_draw', type=str, choices=parameters_choices['type_draw'], default=parameters_default['type_draw'], help=parameters_help['type_draw'])
        subparser.add_argument('--num_images', type=int, default=parameters_default['num_images'], help=parameters_help['num_images'])
        subparser.add_argument('--idx', type=int, default=parameters_default['idx'], help=parameters_help['idx'])
        subparser.add_argument('--model_dir', type=str, default=parameters_default['model_dir'], help=parameters_help['model_dir'])

    # Parse arguments
    parser = parser.parse_args()

    return parser
