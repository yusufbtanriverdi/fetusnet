import sys
import yaml
import ast
from types import SimpleNamespace
from utils import flatten_namespace_to_dict

# Hide the top-level mode mappings from the main options list
skipped_keys = ["train", "test", "prepare", "preprocess", "rotate", "presplit", "help", "generate", 
                "script_conceptual", "script_stats", "script_ELA", "test_loaders", "test_pipe", 
                "interactive_plot", "interactive_game"
                ]

# ==========================================
# BASE STRUCTURE
# ==========================================
class RecursiveNamespace(SimpleNamespace):
    @staticmethod
    def map_entry(entry):
        if isinstance(entry, dict):
            return RecursiveNamespace(**entry)
        return entry

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        for key, val in kwargs.items():
            if type(val) == dict:
                setattr(self, key, RecursiveNamespace(**val))
            elif type(val) == list:
                setattr(self, key, list(map(self.map_entry, val)))

def load_yaml(filepath):
    try:
        with open(filepath, 'r') as f:
            data = yaml.safe_load(f)
            return RecursiveNamespace(**data) if data else RecursiveNamespace()
    except FileNotFoundError:
        return RecursiveNamespace()

# ==========================================
# HELPER TO DYNAMICALLY UPDATE & PRINT HELP
# ==========================================
def update_nested_namespace(ns, key_path, value_str):
    """Traverses the RecursiveNamespace and overrides or injects a nested value."""
    # Convert incoming CLI strings to native Python types (e.g. "100" -> 100, "true" -> True)
    try:
        # ast.literal_eval safely handles booleans, numbers, lists, etc.
        value = ast.literal_eval(value_str)
    except (ValueError, SyntaxError):
        value = value_str # Fallback to pure string if it can't be parsed

    parts = key_path.split('.')
    current = ns
    for part in parts[:-1]:
        if not hasattr(current, part):
            setattr(current, part, RecursiveNamespace())
        current = getattr(current, part)
        
    setattr(current, parts[-1], value)

def print_custom_help(config_ns, help_ns):
    """Generates a clean help overview mapping parameters to descriptions."""
    print("\n" + "="*60)
    print(" FetusNet Command Line Manual")
    print("="*60)
    print("\n📝 Dynamic Overrides Syntax:")
    print("  python main.py --mode=train --train_.epochs=100")
    print("  python main.py --prepocessing_.params.gt++.params.desired_size=\"[128,128,128]\"\n")
    print("⚙️ Configured Parameter Tree & Descriptions:")
    
    flat_config = flatten_namespace_to_dict(config_ns)
    flat_help = flatten_namespace_to_dict(help_ns)
    
    for key, current_val in flat_config.items():
        if key in skipped_keys:
            continue
        description = flat_help.get(key, "No documentation provided for this field.")
        print(f"  --{key:<55} {description}")
        print(f"    [Current Value: {current_val}]\n")
    print("="*60)

# ==========================================
# OVERWRITE & MODE SETUP FUNCTION
# ==========================================

def setup_config(default_path="net/config/default.yaml", help_path="net/config/help.yaml"):
    # 1. Load initial configurations
    config = load_yaml(default_path)
    help_cfg = load_yaml(help_path)
    
    # 2. Intercept and loop through sys.argv to override configuration properties
    # Supports both '--key=value' and '--key value' structures
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        arg = args[i]
        if arg.startswith('--'):
            if '=' in arg:
                key_part, val_part = arg.split('=', 1)
                key_path = key_part.lstrip('-')
                update_nested_namespace(config, key_path, val_part)
            else:
                key_path = arg.lstrip('-')
                if i + 1 < len(args) and not args[i+1].startswith('--'):
                    update_nested_namespace(config, key_path, args[i+1])
                    i += 1
                else:
                    # Boolean flag fallback if no value is assigned to the argument
                    update_nested_namespace(config, key_path, "True")
        i += 1

    # 3. Intercept direct help flags
    if '-h' in sys.argv or '--help' in sys.argv:
        config.mode = "help"

    # 4. Resolve the active execution mode (Number vs String mapping)
    # Collect all available integer mapping rules embedded at the root level of default.yaml
    modes_dict = {k: v for k, v in config.__dict__.items() if isinstance(v, int) and k != 'mode'}
    
    # Check if the user specified an integer via the CLI or YAML defaults
    current_mode_raw = config.mode
    
    if str(current_mode_raw).isdigit() or isinstance(current_mode_raw, int):
        target_id = int(current_mode_raw)
        # Find the string name associated with the input number
        resolved_mode = next((name for name, idx in modes_dict.items() if idx == target_id), "help")
        config.mode = resolved_mode
    
    # 5. Route to Help Terminal Screen if active mode is resolved as 'help'
    if config.mode == "help" or config.mode not in skipped_keys:
        print_custom_help(config, help_cfg)
        sys.exit(0)
        
    return config

# ==========================================
# RUN PIPELINE
# ==========================================

# if __name__ == "__main__":
#     # Resolve values, handle help context, map modes
#     cfg = setup_config()
    
#     # Execute with clean native dot-notation syntax!
#     print(f"\n🚀 Pipeline initialized successfully!")
#     print(f"Active Mode Name: {cfg.mode}")
#     print(f"Target Hardware:  {cfg.device}")
    
#     if cfg.mode == "train":
#         print(f"Running training sequence on architecture: {cfg.train_.architecture}")
#         print(f"Epoch limit set to: {cfg.train_.epochs}")
        
#     elif cfg.mode == "preprocess":
#         # Safe dot notation access, even down into sections containing symbols like 'gt++'
#         # Dict syntax fallback is avoided completely
#         tgt_dimensions = cfg.prepocessing_.params.__dict__['gt++'].params.desired_size
#         print(f"Executing Preprocessing sequence.")
#         print(f"Reshaping volume space to: {tgt_dimensions}")