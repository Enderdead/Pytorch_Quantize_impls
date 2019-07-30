
from copy import deepcopy
from utils.tools import flat_net
import numpy as np
import functools


NB_BITS_MEAN = 2
START_BITS = 8
MAX= 0.82

def init_conf(flat_model):
    """
    Generate init conf
    """
    model_curr_conf = [START_BITS for _ in range(len(flat_model))]
    return model_curr_conf

def condition(model):
    """
    La condition à remplir pour arreté la compression
    """
    mean = functools.reduce(lambda x ,y : x+y , model)/len(model)
    return mean<NB_BITS_MEAN


def generate_next_step(flat_model, model_curr_conf):
    """
    Generate next conf to try
    """
    todo_quants = [deepcopy(model_curr_conf) for _ in range(len(flat_model))]
    
    for index, conf in enumerate(todo_quants):
        conf[index] -=1
    
    todo_quants = list(filter( lambda x: not 1 in x ,todo_quants))
    return todo_quants

def select_best(dataframe, flat_model):
    """
    Return best index
    """
    vect_weight = list()
    for layer in flat_model:
        vect_weight.append(len(layer.weight.data.cpu().detach().numpy().reshape(-1)  ))

    vect_weight = np.array(vect_weight)
    res  = (MAX-dataframe["value"])* (dataframe["user_attrs"]*vect_weight).sum(1)
    return res.idxmin()