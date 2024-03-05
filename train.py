import importlib 
import sys 

directory = sys.argv[1]

def load_model(directory):
    directory_list = directory.rsplit(".", 1)
    module = importlib.import_module(directory_list[0])
    function = getattr(module, directory_list[1])
    return function

fun = load_model(directory)