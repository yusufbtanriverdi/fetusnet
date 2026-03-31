import argparse
import json
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'default.json')
HELP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'help.json')
CHOICES_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'choices.json')


def load_json_config(path):
    # Loads json config file
    with open(path, 'r') as f:
        return json.load(f)


def load_config(default_path=DEFAULT_CONFIG_PATH, user_path=None):
    # Load defaults first
    with open(default_path, 'r') as f:
        config = json.load(f)

    # If user config provided and exists, update defaults with it
    if user_path and os.path.exists(user_path):
        with open(user_path, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)

    return config


parameters_help = load_json_config(HELP_CONFIG_PATH)
parameters_choices = load_json_config(CHOICES_CONFIG_PATH)


def print_help(*args, **kwargs):
    print("Available modes:")
    for mode in ['train', 'test', 'prepare', 'rotate', 'presplit', 'help', 'generate', 'test_loaders']:
        print(f"  {mode:16} {parameters_help.get(mode, '')}")


def add_common_args(subparser: argparse.ArgumentParser, defaults: dict):
    """Adds all common arguments to a subparser"""
    subparser.add_argument('--root', type=str, default=defaults.get('root'), help=parameters_help['root'])
    subparser.add_argument('--sys', type=str, default=defaults.get('sys'), help=parameters_help['sys'])
    subparser.add_argument('--raw_dir', type=str, default=defaults.get('raw_dir'), help=parameters_help['raw_dir'])
    subparser.add_argument('--mdir', type=str, default=defaults.get('mdir'), help=parameters_help['mdir'])
    subparser.add_argument('--wandbpro', type=str, default=defaults.get('wandbpro'), help=parameters_help['wandbpro'])
    subparser.add_argument('--device', type=str, default=defaults.get('device'), help=parameters_help['device'])

    subparser.add_argument('--split', type=str, default=defaults.get('split'), help=parameters_help['split'])
    subparser.add_argument('--n_split', '-n', type=int, default=defaults.get('n_split'), help=parameters_help['n_split'])
    subparser.add_argument('--test_patients', nargs='+', type=int, help=parameters_help['test_patients'])
    subparser.add_argument('--master_df', type=str, default=defaults.get('master_df'), help=parameters_help['master_df'])

    # Rotation related
    subparser.add_argument('--desired_size', type=int, default=defaults.get('desired_size'), help=parameters_help['desired_size'])
    subparser.add_argument('--desired_spacings', type=int, default=defaults.get('desired_spacings'), help=parameters_help['desired_spacings'])

    subparser.add_argument('--train_ds', nargs='+', type=str, default=defaults.get('train_ds'), help=parameters_help['train_val_ds'])
    subparser.add_argument('--test_ds', nargs='+', type=str, default=defaults.get('test_ds'), help=parameters_help['test_ds'])
    subparser.add_argument('--generate', type=str, default=defaults.get('generate'), choices=parameters_choices['generate'], help=parameters_help['generate'])
    subparser.add_argument('--g_alpha', type=float, default=defaults.get('g_alpha'), help=parameters_help['g_alpha'])
    subparser.add_argument('--g_eps', type=int, default=defaults.get('g_eps'), help=parameters_help['g_eps'])

    subparser.add_argument('--prefix', type=str, default=defaults.get('prefix'), help=parameters_help['prefix'])
    subparser.add_argument('--use_wandb', action='store_true', default=defaults.get('use_wandb'), help=parameters_help['use_wandb'])
    subparser.add_argument('--iter_folds', '-ifs', nargs='+', type=int, default=defaults.get('iter_folds'), help=parameters_help['iter_folds'])
    subparser.add_argument('--resume', action='store_true', default=defaults.get('resume'), help=parameters_help['resume'])
    subparser.add_argument('--seed', type=int, default=defaults.get('seed'), help=parameters_help['seed'])

    subparser.add_argument('--rescale', action='store_true', default=defaults.get('rescale'), help=parameters_help['rescale'])
    subparser.add_argument('--lmks', type=str, nargs='+', default=defaults.get('lmks'), help=parameters_help['lmks'])

    subparser.add_argument('--batch_size_train', '-bs', type=int, default=defaults.get('batch_size_train'), help=parameters_help['batch_size_train'])
    subparser.add_argument('--batch_size_val', type=int, default=defaults.get('batch_size_val'), help=parameters_help['batch_size_val'])
    subparser.add_argument('--batch_size_test', type=int, default=defaults.get('batch_size_test'), help=parameters_help['batch_size_test'])
    subparser.add_argument('--num_workers', type=int, default=defaults.get('num_workers'), help=parameters_help['num_workers'])

    subparser.add_argument('--architecture', '-a', type=str, default=defaults.get('architecture'), help=parameters_help['architecture'])

    subparser.add_argument('--epochs', '-ep', type=int, default=defaults.get('epochs'), help=parameters_help['epochs'])
    subparser.add_argument('--num_fts', type=int, default=defaults.get('num_fts'), help=parameters_help['num_fts'])
    subparser.add_argument('--optimizer', '-optim', type=str, default=defaults.get('optimizer'), choices=parameters_choices['optimizer'], help=parameters_help['optimizer'])
    subparser.add_argument('--learning_rate', '-lr', type=float, default=defaults.get('learning_rate'), help=parameters_help['learning_rate'])
    subparser.add_argument('--lr_momentum', '-m', type=int, default=defaults.get('lr_momentum'), help=parameters_help['lr_momentum'])

    subparser.add_argument('--reduction', type=str, choices=parameters_choices['reduction'], default=defaults.get('reduction'), help=parameters_help['reduction'])
    subparser.add_argument('--loss', '-l', type=str, default=defaults.get('loss'), help=parameters_help['loss'])

    subparser.add_argument('--checkpoint_dir', type=str, default=defaults.get('checkpoint_dir'), help=parameters_help['checkpoint_dir'])
    subparser.add_argument('--use_model', type=str, default=defaults.get('use_model'), help=parameters_help['use_model'], choices=parameters_choices['use_model'])
    subparser.add_argument('--radius_eval', type=int , default=defaults.get('radius_eval'), help=parameters_help['radius_eval'])
    subparser.add_argument('--radius_num', type=int, default=defaults.get('radius_num'), help=parameters_help['radius_num'])
    subparser.add_argument('--detector', type=str, default=defaults.get('detector'), help=parameters_help['detector'])
    subparser.add_argument('--progress_bar', action='store_true', default=defaults.get('progress_bar'), help=parameters_help['progress_bar'])
    subparser.add_argument('--check_visibility', action='store_true', default=defaults.get('check_visibility', False), help=parameters_help.get('check_visibility', 'Check visibility of landmarks'))
    subparser.add_argument('--save_targets', action='store_true', default=defaults.get('save_targets', False), help=parameters_help.get('save_targets', 'Save target values'))
    subparser.add_argument('--save_outputs', action='store_true', default=defaults.get('save_outputs', False), help=parameters_help.get('save_outputs', 'Save model outputs'))

def parameters_parsing() -> argparse.Namespace:
    """Unified entry point for parameter parsing with JSON config override."""

    # Step 1: Early parse to get --config
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument('--config', type=str, help='Path to JSON config to override defaults')
    early_args, remaining_argv = early_parser.parse_known_args()

    # Step 2: Load config if provided
    defaults = load_config(user_path=early_args.config if early_args.config else None)

    # Step 3: Full parser
    parser = argparse.ArgumentParser(description='FetusNet CLI', parents=[early_parser])

    # Step 4: Subcommands
    subparsers = parser.add_subparsers(dest='mode', metavar='mode', title=parameters_help['mode'])

    train_parser = subparsers.add_parser('train', help=parameters_help['train'])
    test_parser = subparsers.add_parser('test', help=parameters_help['test'])
    prep_parser = subparsers.add_parser('prepare', help=parameters_help['prepare'])
    rotate_parser = subparsers.add_parser('rotate', help=parameters_help['rotate'])
    presplit_parser = subparsers.add_parser('presplit', help=parameters_help['presplit'])
    help_parser = subparsers.add_parser('help', help=parameters_help['help'])
    generate_parser = subparsers.add_parser('generate', help=parameters_help['generate'])
    pipe_parser = subparsers.add_parser('test_loaders', help=parameters_help['test_loaders'])
    help_parser.set_defaults(func=print_help)

    # Step 5: Add arguments with JSON-loaded defaults
    for sub in [train_parser, test_parser, prep_parser, presplit_parser, rotate_parser, generate_parser, pipe_parser]:
        add_common_args(sub, defaults)

    # Step 6: Final parse using updated parser and remaining args
    args = parser.parse_args(remaining_argv)
    args.config_dir = early_args.config
    
    # Step 7: Handle 'help' mode
    if args.mode == 'help':
        print_help()
        exit(0)

    return args


