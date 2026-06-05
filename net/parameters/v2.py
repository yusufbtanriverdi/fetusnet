import argparse
import json
import os

DEFAULT_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'default.json')
HELP_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'help.json')
CHOICES_CONFIG_PATH = os.path.join(os.path.dirname(__file__), 'choices.json')

def load_json_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def load_config(default_path=DEFAULT_CONFIG_PATH, user_path=None):
    with open(default_path, 'r') as f:
        config = json.load(f)
    if user_path and os.path.exists(user_path):
        with open(user_path, 'r') as f:
            user_config = json.load(f)
        config.update(user_config)
    return config

def extract_flat_defaults(config: dict, mode: str) -> dict:
    """
    Adapts the highly nested default.json structure into a flat dictionary 
    tailored to the active CLI subparser mode to avoid key collisions.
    """
    flat = {}
    
    # Global level properties
    flat['device'] = config.get('device', 'cuda')
    flat['prefix'] = config.get('prefix', 'v2')
    flat['resume'] = config.get('resume', False)
    flat['checkpoint_dir'] = config.get('checkpoint_', '')

    # Reproducibility mapping
    repro = config.get('reproducibility_', {})
    flat['base_seed'] = repro.get('model_seed', 42)
    flat['torch_seed'] = repro.get('generator_seed', 42)

    # Dataset & Split mappings
    ds = config.get('dataset_', {})
    flat['master_df'] = ds.get('dataframe', '')
    flat['root'] = ds.get('root', 'DATA/')
    flat['sys'] = ds.get('sys', '')

    split = config.get('split_', {})
    flat['n_split'] = split.get('n_split', 4)
    flat['test_ds'] = split.get('test_ds', [])
    flat['test_patients'] = split.get('test_patients', [])
    flat['train_ds'] = split.get('train_val_ds', [])

    # Target & Landmarks
    flat['lmks'] = config.get('target_', {}).get('lmks', [])

    # Heatmaps
    g_map = config.get('gaussian_heatmap_', {})
    flat['g_alpha'] = g_map.get('alpha', 3)
    flat['g_eps'] = g_map.get('eps', 1e-6)
    flat['g_mask'] = g_map.get('mask', False)
    flat['g_clip'] = g_map.get('clip', False)

    # Loss configurations
    loss_sec = config.get('loss_', {})
    flat['reduction'] = loss_sec.get('reduction', 'mean')
    flat['loss_params'] = loss_sec  # Kept as dictionary to skip manual parsing if unmodified

    # Optimizer defaults based on selected train execution
    train_sec = config.get('train_', {})
    chosen_optim = train_sec.get('optimizer', 'adam')
    flat['optimizer'] = chosen_optim
    
    optim_sec = config.get('optimizer_', {})
    if chosen_optim in optim_sec:
        flat['learning_rate'] = optim_sec[chosen_optim].get('lr', 0.0001)
        flat['lr_momentum'] = optim_sec[chosen_optim].get('momentum', 0.9 if chosen_optim == 'sgd' else 0)

    # Nested Preprocessing extractions
    prep = config.get('prepocessing_', {})
    gt_params = prep.get('params', {}).get('gt++', {}).get('params', {})
    
    # argparse expects single integers for sizes/spacings here
    flat['desired_size'] = gt_params.get('desired_size', [256, 256, 256])[0]
    flat['desired_spacings'] = prep.get('params', {}).get('bspline', {}).get('spacing_', [1.0, 1.0, 1.0])[0]
    flat['mdir'] = prep.get('params', {}).get('gt++', {}).get('model_dir', '')
    flat['raw_dir'] = prep.get('params', {}).get('gt++', {}).get('file_paths', {}).get('data_dir', '')

    # Contextual routing (Handles key collisions cleanly)
    if mode == 'train':
        flat['architecture'] = train_sec.get('architecture', 'resunet3d')
        flat['batch_size_train'] = train_sec.get('batch_size_train', 2)
        flat['epochs'] = train_sec.get('epochs', 50)
        flat['loss'] = train_sec.get('loss', ['softmaxce'])
        flat['mnl'] = train_sec.get('mnl', False)
        flat['lambdas'] = train_sec.get('lambdas', [0.5])
        flat['num_workers'] = train_sec.get('num_workers', 1)
        flat['progress_bar'] = train_sec.get('progress_bar', False)
        flat['rescale'] = train_sec.get('rescale_inputs', True)
        flat['num_fts'] = train_sec.get('num_fts', 32)
    
    elif mode in ['test', 'test_loaders', 'test_pipe', 'plot_3d', 'game_3d']:
        eval_sec = config.get('eval_', {})
        flat['batch_size_test'] = eval_sec.get('batch_size_test', 1)
        flat['use_model'] = eval_sec.get('use_model', 'last')
        flat['radius_eval'] = eval_sec.get('radius_eval', 40)
        flat['radius_num'] = eval_sec.get('radius_num', 40)
        flat['save_outputs'] = eval_sec.get('save_outputs', False)
        flat['save_targets'] = eval_sec.get('save_targets', False)
        flat['progress_bar'] = eval_sec.get('progress_bar', False)
        flat['num_workers'] = config.get('loaders_', {}).get('num_workers', 1)
        
    else:  # Data preprocessing modes fallback
        flat['progress_bar'] = False
        flat['num_workers'] = 1
        flat['epochs'] = gt_params.get('epochs', 100000)

    # Validation configs shared defaults
    val_sec = config.get('val_', {})
    flat['batch_size_val'] = val_sec.get('batch_size_val', 1)
    flat['check_visibility'] = val_sec.get('check_visibility', True)
    flat['detector'] = val_sec.get('detector', 'argmax')

    # Weights and Biases Setup
    wandb_sec = config.get('wandb_', {})
    flat['use_wandb'] = wandb_sec.get('log', False)
    flat['wandbpro'] = wandb_sec.get('wandbpro', 'fetusnetv3')

    return flat

def parse_loss_params(values):
    if values is None:
        return {}
    if len(values) % 2 != 0:
        raise argparse.ArgumentTypeError("loss_params must be key value pairs")
    result = {}
    for k, v in zip(values[::2], values[1::2]):
        if v.lower() == "true": v = True
        elif v.lower() == "false": v = False
        else:
            try: v = int(v)
            except ValueError:
                try: v = float(v)
                except ValueError: pass
        result[k] = v
    return result

parameters_help = load_json_config(HELP_CONFIG_PATH)
parameters_choices = load_json_config(CHOICES_CONFIG_PATH)

def print_help(*args, **kwargs):
    print("Available modes:")
    for mode in parameters_choices['mode']:
        print(f"  {mode:16} {parameters_help.get(mode, '')}")

def add_common_args(subparser: argparse.ArgumentParser, defaults: dict):
    # (Kept identical to yours so your argument configurations don't break)
    subparser.add_argument('--architecture', '-a', type=str, default=defaults.get('architecture'), help=parameters_help['architecture'])
    subparser.add_argument('--base_seed', type=int, default=defaults.get('base_seed'), help=parameters_help['base_seed'])
    subparser.add_argument('--batch_size_test', type=int, default=defaults.get('batch_size_test'), help=parameters_help['batch_size_test'])
    subparser.add_argument('--batch_size_train', '-bs', type=int, default=defaults.get('batch_size_train'), help=parameters_help['batch_size_train'])
    subparser.add_argument('--batch_size_val', type=int, default=defaults.get('batch_size_val'), help=parameters_help['batch_size_val'])
    subparser.add_argument('--check_visibility', action='store_true', default=defaults.get('check_visibility', False), help=parameters_help.get('check_visibility', 'Check visibility of landmarks'))
    subparser.add_argument('--checkpoint_dir', type=str, default=defaults.get('checkpoint_dir'), help=parameters_help['checkpoint_dir'])
    subparser.add_argument('--desired_size', type=int, default=defaults.get('desired_size'), help=parameters_help['desired_size'])
    subparser.add_argument('--desired_spacings', type=int, default=defaults.get('desired_spacings'), help=parameters_help['desired_spacings'])
    subparser.add_argument('--detector', type=str, default=defaults.get('detector'), help=parameters_help['detector'])
    subparser.add_argument('--device', type=str, default=defaults.get('device'), help=parameters_help['device'])
    subparser.add_argument('--epochs', '-ep', type=int, default=defaults.get('epochs'), help=parameters_help['epochs'])
    subparser.add_argument('--g_alpha', type=float, default=defaults.get('g_alpha'), help=parameters_help['g_alpha'])
    subparser.add_argument('--g_eps', type=int, default=defaults.get('g_eps'), help=parameters_help['g_eps'])
    subparser.add_argument('--g_clip', action='store_true', default=defaults.get('g_clip'), help=parameters_help['g_clip'])
    subparser.add_argument('--g_mask', action='store_true', default=defaults.get('g_mask'), help=parameters_help['g_mask'])
    subparser.add_argument('--iter_folds', '-ifs', nargs='+', type=int, default=defaults.get('iter_folds'), help=parameters_help['iter_folds'])
    subparser.add_argument('--lambdas', '-w', nargs='+', type=float, default=defaults.get('lambdas'), help=parameters_help['lambdas'])
    subparser.add_argument('--learning_rate', '-lr', type=float, default=defaults.get('learning_rate'), help=parameters_help['learning_rate'])
    subparser.add_argument('--lmks', type=str, nargs='+', default=defaults.get('lmks'), help=parameters_help['lmks'])
    subparser.add_argument('--loss', '-l', nargs='+', type=str, default=defaults.get('loss'), help=parameters_help['loss'])
    subparser.add_argument('--loss_params', nargs='+', default=defaults.get('loss_params'), help=parameters_help['loss_params'])
    subparser.add_argument('--lr_momentum', '-m', type=int, default=defaults.get('lr_momentum'), help=parameters_help['lr_momentum'])
    subparser.add_argument('--master_df', type=str, default=defaults.get('master_df'), help=parameters_help['master_df'])
    subparser.add_argument('--mdir', type=str, default=defaults.get('mdir'), help=parameters_help['mdir'])
    subparser.add_argument('--mnl', action='store_true', default=defaults.get('mnl', False), help=parameters_help.get('mnl'))
    subparser.add_argument('--n_split', '-n', type=int, default=defaults.get('n_split'), help=parameters_help['n_split'])
    subparser.add_argument('--num_fts', type=int, default=defaults.get('num_fts'), help=parameters_help['num_fts'])
    subparser.add_argument('--num_workers', type=int, default=defaults.get('num_workers'), help=parameters_help['num_workers'])
    subparser.add_argument('--optimizer', '-optim', type=str, default=defaults.get('optimizer'), choices=parameters_choices['optimizer'], help=parameters_help['optimizer'])
    subparser.add_argument('--prefix', type=str, default=defaults.get('prefix'), help=parameters_help['prefix'])
    subparser.add_argument('--progress_bar', action='store_true', default=defaults.get('progress_bar'), help=parameters_help['progress_bar'])
    subparser.add_argument('--radius_eval', type=int , default=defaults.get('radius_eval'), help=parameters_help['radius_eval'])
    subparser.add_argument('--radius_num', type=int, default=defaults.get('radius_num'), help=parameters_help['radius_num'])
    subparser.add_argument('--raw_dir', type=str, default=defaults.get('raw_dir'), help=parameters_help['raw_dir'])
    subparser.add_argument('--reduction', type=str, choices=parameters_choices['reduction'], default=defaults.get('reduction'), help=parameters_help['reduction'])
    subparser.add_argument('--rescale', action='store_true', default=defaults.get('rescale'), help=parameters_help['rescale'])
    subparser.add_argument('--resume', action='store_true', default=defaults.get('resume'), help=parameters_help['resume'])
    subparser.add_argument('--root', type=str, default=defaults.get('root'), help=parameters_help['root'])
    subparser.add_argument('--save_outputs', action='store_true', default=defaults.get('save_outputs', False), help=parameters_help.get('save_outputs'))
    subparser.add_argument('--save_targets', action='store_true', default=defaults.get('save_targets', False), help=parameters_help.get('save_targets'))
    subparser.add_argument('--split', type=str, default=defaults.get('split'), help=parameters_help['split'])
    subparser.add_argument('--sys', type=str, default=defaults.get('sys'), help=parameters_help['sys'])
    subparser.add_argument('--test_ds', nargs='+', type=str, default=defaults.get('test_ds'), help=parameters_help['test_ds'])
    subparser.add_argument('--test_patients', nargs='+', type=int, default=defaults.get('test_patients'), help=parameters_help['test_patients'])
    subparser.add_argument('--torch_seed', type=int, default=defaults.get('torch_seed'), help=parameters_help['torch_seed'])
    subparser.add_argument('--train_ds', nargs='+', type=str, default=defaults.get('train_ds'), help=parameters_help['train_val_ds'])
    subparser.add_argument('--use_model', type=str, default=defaults.get('use_model'), help=parameters_help['use_model'], choices=parameters_choices['use_model'])
    subparser.add_argument('--use_wandb', action='store_true', default=defaults.get('use_wandb'), help=parameters_help['use_wandb'])
    subparser.add_argument('--wandbpro', type=str, default=defaults.get('wandbpro'), help=parameters_help['wandbpro'])

def parameters_parsing() -> argparse.Namespace:
    early_parser = argparse.ArgumentParser(add_help=False)
    early_parser.add_argument('--config', type=str, help='Path to JSON config to override defaults')
    early_args, remaining_argv = early_parser.parse_known_args()
    
    defaults = load_config(user_path=early_args.config if early_args.config else None)
    parser = argparse.ArgumentParser(description='FetusNet CLI', parents=[early_parser])
    
    subparsers = parser.add_subparsers(dest='mode', metavar='mode', title=parameters_help['mode'])
    
    # Subparsers mapping structure
    modes_map = {
        'train': subparsers.add_parser('train', help=parameters_help['train']),
        'test': subparsers.add_parser('test', help=parameters_help['test']),
        'prepare': subparsers.add_parser('prepare', help=parameters_help['prepare']),
        'preprocess': subparsers.add_parser('preprocess', help=parameters_help['preprocess']),
        'presplit': subparsers.add_parser('presplit', help=parameters_help['presplit']),
        'help': subparsers.add_parser('help', help=parameters_help['help']),
        'generate': subparsers.add_parser('generate', help=parameters_help['generate']),
        'test_loaders': subparsers.add_parser('test_loaders', help=parameters_help['test_loaders']),
        'plot_3d': subparsers.add_parser('plot_3d', help=parameters_help['plot_3d']),
        'game_3d': subparsers.add_parser('game_3d', help=parameters_help['game_3d']),
        'script_concept': subparsers.add_parser('script_concept', help=parameters_help['script_concept'])
    }
    
    modes_map['help'].set_defaults(func=print_help)
    
    # Map context-aware defaults for every individual sub-command
    for mode_name, sub_parser in modes_map.items():
        mode_defaults = extract_flat_defaults(defaults, mode_name)
        add_common_args(sub_parser, mode_defaults)

    args = parser.parse_args(remaining_argv)
    
    # If loss_params isn't provided via CLI, it falls back to the JSON dictionary object
    if not isinstance(args.loss_params, dict):
        args.loss_params = parse_loss_params(args.loss_params)
        
    args.config_dir = early_args.config
    if args.mode == 'help':
        print_help()
        exit(0)

    return args