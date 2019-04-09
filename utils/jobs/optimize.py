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
from dataset.mnist import get_mnist
from time import time
import torch
parser = argparse.ArgumentParser()

parser.add_argument("device", type=str, help="Study name to use for this optim")
parser.add_argument("nb_trial", type=int, help="Path to the model")
parser.add_argument("job_index", type=int, help="Path to the model")
parser.add_argument("model_path",type=str)
parser.add_argument("study_name", type=str)
args = parser.parse_args()


spec = importlib.util.spec_from_file_location(os.path.split(args.model_path)[-1], args.model_path)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

device  = args.device
nb_trial = args.nb_trial
job_index = args.job_index


def objectif(trial):
    net = instantiate_model(model.Model, trial=trial)
    data_params, opti_params = get_opti_from_model(model.Model, trial)

    batch_size = data_params.get("batch_size",128)

    valid_ratio = data_params.get("valid_ratio", 0.2)

    max_epoch = data_params.get("epoch", 20)

    learning_rate = opti_params.get("lr", 0.01)

    decay_lr = opti_params.get("decay_lr", 0)

    optimizer = opti_params.get("optimizer", torch.optim.Adam)

    train_set, valid_set, _ = get_mnist(batch_size=batch_size, valid_ratio=valid_ratio, directory="/tmp/MNIST", transform_both=None, transform_train=None, transform_valid=None)

    scheduled_lr = [learning_rate*((1-decay_lr)**index) for index in range(20)]
    scheduled_opti_conf = {}
    for index, lr in enumerate(scheduled_lr):
        scheduled_opti_conf[index] = {"lr":lr}
    

    res = train(net, train_set, valid_set, device=device, save_path="/tmp/Optimize/job_{}".format(job_index), early_stopping=False,
                opti=optimizer, loss=torch.nn.CrossEntropyLoss(), max_epoch=max_epoch,
                static_opti_conf=None, scheduled_opti_conf=scheduled_opti_conf, accuracy_method=Accuracy(10))
    
    return res


study = optuna.create_study(storage="sqlite:///{}.db".format(args.study_name),study_name=args.study_name, load_if_exists=True, direction="maximize",sampler=optuna.samplers.RandomSampler(seed=int(time())//10000+args.job_index))
study.optimize(objectif, n_trials=nb_trial)
