import sys
from cx_Freeze import setup, Executable

# base="Win32GUI" should be used only for Windows GUI app
base = "Win32GUI" if sys.platform == "win32" else None

setup(
    name="TFT-AI",
    version="0.1",
    description="My AI!",
    executables=[Executable("app.py", base=base)],
)