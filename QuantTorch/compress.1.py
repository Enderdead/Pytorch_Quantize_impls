import argparse
import importlib.util
import os.path
import functools
from copy import deepcopy
import configparser
import optuna
from threading import Thread
from train.instantiate import *
from train.train import train
from random import random
import numpy as np
import torch.nn as nn 
from layers.common import QLayer
from multiprocessing import Process, Lock
from train.metrics import Accuracy
from dataset.mnist import get_mnist
import torch
from models.sample import BinMNIST
import pandas as pd 
from utils.tools import flat_net
from time import sleep

optuna.logging.set_verbosity(optuna.logging.ERROR)
"""
parser = argparse.ArgumentParser(description='DNN Compressor')


parser.add_argument('-j','--jobs', type=int, default=4, metavar='j',
                    help='Process to launch for this training')

parser.add_argument('--gpus', type=str, default="",
                    help='gpus used for training - e.g 0,1,3 or 2')


parser.add_argument("-a", "--accuracy", type=float, default=None, help="Set an accuracy threshold to not reach on compress process.")
parser.add_argument("-c", "--compress_ratio", type=float, default=None, help="Set an compress ratio to reach.")


parser.add_argument("-b", "--min_bit", type=float, default=None, help="Set an compress ratio to reach.")
parser.add_argument("-b", "--max_bit", type=float, default=None, help="Set an compress ratio to reach.")


parser.add_argument("study_name", type=str, help="Study name to use for this compression")

parser.add_argument("model", type=str, help=\"""Path to the model script.\n
This script need to have Model class if you want to compress with a uninit network or
a model object loaded at each script call. \""")


#args = parser.parse_args()
"""
1/0
NB_BITS_MEAN = 2.2
START_BITS = 8
THREADS = 8
GPUS = "0,1,2,3"
STUDY_NAME = "test"
MODEL_PATH = None #TODO
# setup work to do per thread
worker = [ {"device":None, "lock": Lock()} for _ in range(THREADS)]



# Setup device 
if GPUS =="":
    # If no gpu setup only use cpu
    for work in worker:
        work["device"] = "cpu"
else:
    gpus = GPUS.split(",")
    for index, work in enumerate(worker):
        work["device"] = "cuda:{}".format(gpus[index%(len(gpus))])


def condition(model):
    """
    La condition à remplir pour arreté la compression
    """
    mean = functools.reduce(lambda x ,y : x+y , model)/len(model)
    return mean<NB_BITS_MEAN


def dataframe_filter(dataframe, round_conf):
    "permet de retourner que les index de la dataframe qui sont dans le round actuel"
    data = dataframe["user_attrs"].values
    index_result = list()
    for index, line in enumerate(round_conf):

        if np.max(np.all(data == line,axis=1)):
            index_result.append(np.argmax(np.all(data == line,axis=1)))
    return index_result

def thread_main(model_path, study_name, conf, device, lock):
    lock.acquire()
    study = optuna.create_study(storage="sqlite:///{}.db".format(study_name), study_name=study_name, load_if_exists=True, direction="maximize")

    def objectif(trial):
        for index, layer  in enumerate(conf):
            trial.set_user_attr("layer_{}".format(index), layer)
        return random()

    study.optimize(objectif, n_trials=1)
    sleep(int(3*random()))
    lock.release()


# On load le model
model = BinMNIST(1,1)

flat_model = flat_net(model, (nn.BatchNorm2d, nn.BatchNorm1d, nn.Linear, nn.Conv2d, QLayer))

main_study = study = optuna.create_study(storage="sqlite:///{}.db".format(STUDY_NAME), study_name=STUDY_NAME, load_if_exists=True, direction="maximize")

def already_done(bits_conf):
    """
    Regarde si la conf a pas déjà été faite
    """
    try:
        confs = main_study.trials_dataframe()["user_attrs"].values
    except (TypeError, KeyError):
        return False
    try:
        res =  np.any(np.all(bits_conf == confs, axis=1))
    except np.AxisError:
        return False
    return res 

def launch(bits_conf, model_path, study_name):
    """
    Permet de lancer la configuration sur un thread ou d'attendre
    """
    # On va chercher un worker de libre
    worker_find = -1
    while worker_find == -1:
        for index, work in enumerate(worker):
            if work["lock"].acquire(block=False):
                worker_find = index
                break
        if worker_find == -1:
            sleep(1)
    print("launch on ", worker_find)
    worker[worker_find]["lock"].release()
    Process(target=thread_main, args=(None, study_name, bits_conf, worker[worker_find]["device"], worker[worker_find]["lock"])).start()


def wait_all():
    "attend tout les threads"
    for work in worker:
        work["lock"].acquire()
        work["lock"].release()

model_curr_conf = [START_BITS for _ in range(len(flat_model))]

while not condition(model_curr_conf):
    # Compute all one layer bis decrease.
    todo_quants = [deepcopy(model_curr_conf) for _ in range(len(flat_model))]
    
    for index, conf in enumerate(todo_quants):
        conf[index] -=1
    
    for conf in todo_quants:
        if already_done(conf):
            print("Deja fait : ", conf)

            continue
        print("On va faire : ", conf)
        launch(conf, model, STUDY_NAME)
    wait_all()

    round_index = dataframe_filter(main_study.trials_dataframe(), todo_quants)
    while len(round_index)!=len(flat_model):
        print("Not enought result got on this round ")
        sleep(1)
        round_index = dataframe_filter(main_study.trials_dataframe(), todo_quants)

    # get best line here : 
    result_index = dataframe_filter(main_study.trials_dataframe(), todo_quants)
    round_result = main_study.trials_dataframe()["value"][result_index]  
    print("round_result ",round_result)
    print("result_index : ", result_index)
    best = list(main_study.trials_dataframe()["user_attrs"].iloc[round_result.idxmax()])
    print("best is ", best)
    model_curr_conf = best
#thread_main(model, "test", 0)



