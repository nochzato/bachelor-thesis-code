import glob
import os


def prepare_dir(dir):
    create_dir(dir)
    clear_dir(dir)


def create_dir(dir):
    if not os.path.exists(dir):
        os.makedirs(dir)


def clear_dir(dir):
    for f in glob.glob(dir + "/*"):
        os.remove(f)
