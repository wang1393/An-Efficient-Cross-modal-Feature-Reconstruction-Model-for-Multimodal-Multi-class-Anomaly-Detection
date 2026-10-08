"""Load a Python experiment config and explicit dotted-key overrides."""
import ast
import importlib
from pathlib import Path
import time

def get_cfg(args):
    path = args.cfg_path.replace('\\', '/')
    module = path[:-3] if path.endswith('.py') else path
    module = module.replace('/', '.')
    cfg = importlib.import_module(module).cfg()
    for key, value in vars(args).items():
        if key != 'opts':
            setattr(cfg, key, value)
    cfg.cfg_path = module
    cfg.task_start_time = time.perf_counter()
    for option in args.opts or []:
        if '=' not in option:
            raise ValueError(f'Expected path.key=value, got {option!r}')
        key, value = option.split('=', 1)
        try:
            value = ast.literal_eval(value)
        except (ValueError, SyntaxError):
            pass
        obj = cfg
        parts = key.split('.')
        for part in parts[:-1]:
            obj = obj[part] if isinstance(obj, dict) else getattr(obj, part)
        if isinstance(obj, dict):
            obj[parts[-1]] = value
        else:
            setattr(obj, parts[-1], value)
    if cfg.mode == 'test' and not cfg.model.kwargs.get('checkpoint_path') and not cfg.trainer.resume_dir:
        raise ValueError('Test mode requires model.kwargs.checkpoint_path or trainer.resume_dir.')
    return cfg
