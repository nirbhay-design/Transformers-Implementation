import importlib 
import sys 
import torch

def load_model(directory):
    directory_list = directory.rsplit(".", 1)
    module = importlib.import_module(directory_list[0])
    function = getattr(module, directory_list[1])
    return function

def test_model(directory, arg, device_name):
    model_name = directory.split('.')[1]
    model_fun = load_model(directory)
    print(model_fun.__name__)
    device = torch.device(device_name)

    model = model_fun(*arg.get(model_name, arg['Default']))
    tsr = torch.rand(2,3,224,224, device=device)
    model = model.to(device)

    out = model(tsr)
    if isinstance(out, tuple) or isinstance(out, list):
        for i in out:
            print(i.shape)
    elif isinstance(out, dict):
        for key, value in out.items():
            print(f"{key}: {value.shape}")
    else:
        print(out.shape)

if __name__ == "__main__":
    directory = sys.argv[1]
    device_name = "cuda:0"
    args = {
        "swinIR": [224],
        "DeTR": [10],
        "Default": [224, 1000]
    }
    test_model(directory, args, device_name)