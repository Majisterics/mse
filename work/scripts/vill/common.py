import os


SCRIPT_PATH = os.path.abspath(__file__)


def cwd_work() -> None:
    """
    Changes the current working directory to the
    work directory of the standard project structure.
    """
    os.chdir('/Users/vill/Major2023/SoftwareEngineering/work')
