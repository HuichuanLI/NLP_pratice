import os


def get_module_path(path):
    return os.path.normpath(os.path.join(os.getcwd(), os.path.dirname(__file__), path))


CRF_MODEL_PATH = get_module_path('pd1998_msra_10w')
