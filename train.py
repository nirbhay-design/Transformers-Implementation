import importlib 
import sys 
import torch
import torch.nn as nn
import torch.nn.functional as F

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

def calculate_acc(scores, labels):
    num_samples = labels.shape[0]
    scores = F.softmax(scores, dim = 1)
    preds, _  = scores.max(dim = 1)
    accuracy = (preds == labels).sum() / num_samples
    return accuracy

def train_cls(
        model,
        dataset,
        optimizer,
        lossfunction,
        epochs,
        return_logs,
        device
):
    model = model.to(device)

    for epoch in range(epochs):
        for idx, (image, labels) in enumerate(dataset):
            image = image.to(device)
            labels = labels.to(device)

            scores = model(image)
            loss = lossfunction(scores, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            cur_train_acc = calculate_acc(scores, labels)
            print(f"current_train_acc: {cur_train_acc}")

            if return_logs:
                print(f"{idx}/{len(dataset)}")

    return model

@torch.no_grad()
def test_cls(
        model,
        dataset,
        device
):
    pass    



if __name__ == "__main__":
    directory = sys.argv[1]
    device_name = "cuda:0"
    args = {
        "swinIR": [224],
        "DeTR": [10],
        "Default": [224, 1000]
    }
    test_model(directory, args, device_name)