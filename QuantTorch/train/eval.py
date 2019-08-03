import torch
from ..train.metrics import Accuracy


def eval_model(model, valid_data, accuracy_method, device="cpu",):
    model = model.to(device)
    model.eval()   
    mean_valid_accuracy = 0.0
    with torch.no_grad():
        curr_batch = 0
        for x,y in valid_data:
            curr_batch += 1
            # Switch params
            x = x.to(device)
            y = y.to(device)

            # Model forward
            outputs = model(x)

            batch_accuracy = accuracy_method(outputs, y)
            mean_valid_accuracy =( (curr_batch-1)*mean_valid_accuracy*valid_data.batch_size + valid_data.batch_size*batch_accuracy  ) /  (curr_batch*valid_data.batch_size)
    
    return mean_valid_accuracy