import os.path
import sys

from cx_Freeze import setup, Executable

base = 'Win32GUI' if sys.platform == 'win32' else None

setup(
    name='TFT-AI',
    version="0.1",
    description="My TFT",
    executables=[Executable(os.path.join(os.curdir, "app.py"), base=base)]
)