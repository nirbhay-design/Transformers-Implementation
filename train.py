import importlib 
import sys 
import torch

directory = sys.argv[1]
device_name = sys.argv[2]
args = sys.argv[3:]
args = list(map(lambda x: int(x), args))

def load_model(directory):
    directory_list = directory.rsplit(".", 1)
    module = importlib.import_module(directory_list[0])
    function = getattr(module, directory_list[1])
    return function

model_fun = load_model(directory)

device = torch.device(device_name)

model = model_fun(*args)
tsr = torch.rand(2,3,224,224, device=device)
model = model.to(device)

out = model(tsr)
if isinstance(out, tuple):
    for i in out:
        print(i.shape)
else:
    print(out.shape)