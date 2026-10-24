"""`--config run.yaml` for long commands. Each command reads its own section, keys are its flag names; flags on the command line win."""
import argparse
import sys
from pathlib import Path

import yaml

from osu_auto.paths import ROOT


def parse_args(p: argparse.ArgumentParser, section: str, path_keys: tuple[str, ...] = ()) -> argparse.Namespace:
    """Relative paths under `path_keys` in the YAML resolve from the repo root."""
    p.add_argument("--config", help=f"YAML file whose `{section}:` section sets defaults for these flags")
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config")
    known, _ = pre.parse_known_args(sys.argv[1:])
    if known.config:
        values = (yaml.safe_load(Path(known.config).read_text()) or {}).get(section) or {}
        actions = {a.dest: a for a in p._actions}
        defaults = {}
        for key, value in values.items():
            dest = key.replace("-", "_")
            if dest not in actions:
                p.error(f"{known.config}: unknown key {key!r} in {section}")
            takes_list = actions[dest].nargs in ("*", "+") or isinstance(actions[dest], argparse._AppendAction)
            if takes_list and not isinstance(value, list):
                value = [value]
            if key in path_keys:
                value = [_resolve(v) for v in value] if isinstance(value, list) else _resolve(value)
            defaults[dest] = value
        p.set_defaults(**defaults)
    return p.parse_args()


def _resolve(value: str) -> str:
    path = Path(value)
    return str(path if path.is_absolute() else ROOT / path)
