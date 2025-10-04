import shutil
import pathlib


[shutil.rmtree(p) for
 p in pathlib.Path('src').rglob('__pycache__')]