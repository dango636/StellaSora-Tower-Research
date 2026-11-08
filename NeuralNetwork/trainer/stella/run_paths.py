"""Access portable run references without rewriting frozen JSON identities."""
from pathlib import Path, PurePosixPath


def relative_path(value):
    text = str(value).replace('\\', '/')
    path = PurePosixPath(text)
    if path.is_absolute() or '..' in path.parts or any(':' in part for part in path.parts):
        raise ValueError('Run reference must be a confined relative path: ' + text)
    return path


def run_path(root, reference):
    root = Path(root).resolve()
    path = root.joinpath(*relative_path(reference).parts)
    if not path.resolve().is_relative_to(root):
        raise ValueError('Run reference escapes run directory: ' + str(reference))
    return path


def portable_basename(reference):
    return relative_path(reference).name
