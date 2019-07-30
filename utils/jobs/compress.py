#!/usr/bin/python3
import os
import sys
import optuna

sys.path.insert(0, os.path.join(sys.path[0],'./../..'))
import argparse
import importlib.util
import os.path
import configparser
from multiprocessing import Process
from train.instantiate import *
from train.train import train
from train.metrics import Accuracy
from utils.tools import flat_net, replace_module
from dataset.mnist import get_mnist
from time import time
import torch
parser = argparse.ArgumentParser()

parser.add_argument("device", type=str, help="Study name to use for this optim")
parser.add_argument("job_index", type=int)
parser.add_argument("model_path",type=str)
parser.add_argument("study_name", type=str)
parser.add_argument("compressor_profil", type=str)
parser.add_argument("compress_config", type=str)
args = parser.parse_args()


# Load model
spec_model = importlib.util.spec_from_file_location(os.path.split(args.model_path)[-1], args.model_path)
modellib = importlib.util.module_from_spec(spec_model)
spec_model.loader.exec_module(modellib)

model = modellib.model

# Load convertor
spec_conv = importlib.util.spec_from_file_location(os.path.split(args.compressor_profil)[-1], args.compressor_profil)
convlib = importlib.util.module_from_spec(spec_conv)
spec_conv.loader.exec_module(convlib)
convertor = convlib.convertor
class_to_convert = tuple(convlib.class_converted)
try:
    after_epoch = convlib.after_epoch
except AttributeError:
    after_epoch = None
# Load config 
layer_bitwidth = [int(element) for element in args.compress_config.split(",")]


# Convert init model with good layers !
layers = flat_net(model, class_to_convert)
for index, layer in enumerate(layers):
    new_layer = convertor(layer, layer_bitwidth[index])
    replace_module(model, layer, new_layer)


# On fait l'entrainement
def objectif(trial):
    for index, layer  in enumerate(layer_bitwidth):
        trial.set_user_attr("layer_{}".format(index), layer)

    result = train(model, convlib.train_data, convlib.valid_data, device=args.device, save_path=None, early_stopping=False,
                opti=torch.optim.Adam, loss=torch.nn.CrossEntropyLoss(), max_epoch=convlib.max_epoch,
                static_opti_conf=convlib.static_opti_conf, scheduled_opti_conf=None, after_update=None, before_update=None, after_epoch=after_epoch, accuracy_method=convlib.accuracy_method, verbose=False)
    return result

study = optuna.create_study(storage="sqlite:///{}.db".format(args.study_name),study_name=args.study_name, load_if_exists=True, direction="maximize",sampler=optuna.samplers.RandomSampler(seed=int(time())//10000+args.job_index))
study.optimize(objectif, n_trials=1)
