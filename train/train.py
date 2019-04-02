import os
import sys
sys.path.insert(0, os.path.abspath('..'))
import torch
from train.early_stopping import EarlyStopping
from progress.bar import ChargingBar
from copy import deepcopy
from train.metrics import Accuracy
_EXTERN_PARAM  = ["patience"]

def _update_optimizer(optimizer, params):
    if params is None: return
    for key, value in params:
        if key in _EXTERN_PARAM:
            continue
        for param_group in optimizer.param_groups:
            param_group[key] = value

def train(model, train_data, valid_data, device="cpu", save_path=None, early_stopping=False,
            opti=torch.optim.Adam, loss=torch.nn.CrossEntropyLoss(), max_epoch=20,
            static_opti_conf=None, scheduled_opti_conf=None, after_update=None, before_update=None, accuracy_method=None, verbose=False):


    if static_opti_conf is None and scheduled_opti_conf is None:
        raise RuntimeError("No opti conf given !")

    # Move model to good device
    model = model.to(device)

	# Instance optimizer and loss
    init_conf = static_opti_conf if not static_opti_conf is None else scheduled_opti_conf.get(0,dict())
    init_conf = deepcopy(init_conf)
    for element in _EXTERN_PARAM: init_conf.pop(element) # remove patience term
    optimizer = opti(model.parameters(),**init_conf)
    loss = loss.to(device)

	# Init early stopping if needed 
    if early_stopping:
        if not static_opti_conf is None:
            early_control = EarlyStopping(static_opti_conf.get("patience", 7),verbose=True)
        else:
            early_control = EarlyStopping(static_opti_conf.get(0,{"patience": 7}).get("patience", 7),verbose=True)


    mean_train_loss = 0
    mean_train_accuracy = 0

    mean_valid_loss = 0
    mean_valid_accuracy = 0

    for curr_epoch in range(max_epoch):
        print("epoch : ", curr_epoch)
        if not scheduled_opti_conf is None:
            _update_optimizer(optimizer, scheduled_opti_conf.get(curr_epoch, None))
        #Train part
        optimizer.zero_grad()
        model.train()
        curr_batch = 0
        for x,y in train_data:
            curr_batch += 1
            # Switch params
            x = x.to(device)
            y = y.to(device)

            # Model forward
            outputs = model(x)

            optimizer.zero_grad()

            # Loss backward
            loss_val = loss(outputs,y)

            mean_train_loss = (mean_train_loss*((curr_batch-1)*train_data.batch_size) +  train_data.batch_size*loss_val.item())/(curr_batch*train_data.batch_size)
            # accuracy if needed
            if not accuracy_method is None:
                mean_train_accuracy =( (curr_batch-1)*train_data.batch_size*mean_train_accuracy + train_data.batch_size*accuracy_method(outputs, y)   ) /  (curr_batch*train_data.batch_size)

            loss_val.backward()

            # Do before methods
            if "before_update" in list(model.__class__.__dict__.keys()):
                model.before_update()
            if not before_update is None:
                before_update()

            optimizer.step()

            # Do after methods
            if "after_update" in list(model.__class__.__dict__.keys()):
                model.after_update()
            if not after_update is None:
                after_update()

        # Eval part
        model.eval()   
        lol  =0
        with torch.no_grad():
            curr_batch = 0
            for x,y in valid_data:
                curr_batch += 1
                # Switch params
                x = x.to(device)
                y = y.to(device)

                # Model forward
                outputs = model(x)

                # Loss backward
                loss_val = loss(outputs,y)
                mean_valid_loss = (mean_valid_loss*((curr_batch-1)*valid_data.batch_size) +  valid_data.batch_size*loss_val.item())/(curr_batch*valid_data.batch_size)

                # accuracy if needed
                if not accuracy_method is None:
                    batch_accuracy = accuracy_method(outputs, y)/valid_data.batch_size
                    mean_valid_accuracy =( (curr_batch-1)*mean_valid_accuracy*valid_data.batch_size + valid_data.batch_size*batch_accuracy  ) /  (curr_batch*valid_data.batch_size)

        if early_stopping:
            if  not accuracy_method is None: 
                early_control(model,accuracy=mean_valid_accuracy ) 
            else:
                early_control(model,loss=mean_valid_loss )

            if early_control.early_stop:
                break      
    
    if early_stopping:
        if early_control.early_stop:
            #TODO Reload best model
            pass

    return model


from models.sample import BinMNIST
from dataset.mnist import *

if __name__ == "__main__":
    a = BinMNIST(28*28,10,2048)
    train_set, valid_set, _ = get_mnist()

    train(a, train_set, valid_set, device="cuda:0",static_opti_conf={'lr':0.01, "patience":7}, early_stopping=True, accuracy_method=Accuracy(10))