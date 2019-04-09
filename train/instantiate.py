import os
import sys
sys.path.insert(0, os.path.abspath('..'))

from utils.parameters import *
from utils.tools import iterable
from warnings import simplefilter, warn
from random import randint
import optuna
simplefilter("always",RuntimeWarning)


class BinMNIST():
    static_params = [{'coco':1},{'coco':5}]
    vars_params = [[DiscretParameter("num_units",1, 1024, 2048),],[DiscretParameter("num_units", 2, 1024, 2048),]]
    def __init__(self, num_units=1, coco=1):
        print("constructeur avec : ", num_units, coco)
        


def instantiate_model(model, trial=None, constants=None, vars=None):
    """
        This methode instanciate a model with custom params 
    """
    if not model.__dict__.get('static_params', None) is None and not constants is None:
        warn("Static parameters given from two differents sources ! (Take the model's static_params instead of constants kwargs", RuntimeWarning)
        
    if not model.__dict__.get('vars_params', None) is None and not vars is None:
        warn("Static parameters given from two differents sources ! (Take the model's static_params instead of constants kwargs", RuntimeWarning)
    
    selected_constant = model.__dict__.get('static_params', constants)
    selected_vars     = model.__dict__.get('vars_params', vars)

    if not iterable(selected_constant):
        raise RuntimeError("Static parameters not a iterable !")

    if not iterable(selected_vars):
        raise RuntimeError("dynamic parameters not a iterable !")


    if isinstance(selected_constant,(list,tuple)):
        constant_dim = 2
        for index, element in enumerate(selected_constant):
            if not isinstance(element, dict):
                raise RuntimeError("All static parameters version must be put in dict. Expected a dict at index : {}".format(index))
        if len(selected_constant)==1:
            constant_dim = 1
            selected_constant = selected_constant[0]

    elif isinstance(selected_constant, dict):
        constant_dim = 1
    else:
        raise RuntimeError("Static parameters must be store on dict or list of dict !")

    if isinstance(selected_vars,(list,tuple)):
        if isinstance(selected_vars[0], (list, tuple)):
            vars_dim = 2
            for config in selected_vars:
                for index, element in enumerate(config):
                    if not isinstance(element, Parameter):
                        raise RuntimeError("Element  : {} in dynamic params isn't a Parameter object  (index {})!".format(element, index))
            if len(selected_vars)==1:
                vars_dim = 1
                selected_vars = selected_vars[0]
        else:
            for index, element in enumerate(selected_vars):
                if not isinstance(element, Parameter):
                    raise RuntimeError("Element  : {} in dynamic params isn't a Parameter object  (index {})!".format(element, index))
            vars_dim = 1
            

    if vars_dim!=constant_dim:
        raise RuntimeError("Static parameters and dynamics parameter have differents dim. Expected {} (static) equals to {} (dynamic)".format(constant_dim, vars_dim))

    #if not trial is None and vars_dim!=1:
    #    warn("Optuna don't suport mutli distribution, only the first will be use !",RuntimeWarning)
    #    selected_constant= selected_constant[0]
    #    selected_vars = selected_vars[0]
    #    vars_dim = 1
    #    constant_dim = 1

    if (vars_dim!=1):
        if len(selected_constant)!=len(selected_vars):
            raise RuntimeError("Static parameter list and Dynamic parameters list must have the same lenght ! Expected {}={}".format(len(selected_constant), len(selected_vars)))
        rand_idx = randint(0, len(selected_constant)-1)
        final_constant = selected_constant[rand_idx]
        final_vars     = selected_vars[rand_idx]
    else:
        final_constant =  selected_constant
        final_vars = selected_vars


    if trial is None:
        # Get dynamics params:
        final_vars_kwargs = dict()
        for parameter in final_vars:
            final_vars_kwargs[parameter.name] = parameter.apply()
        return model(**{**final_constant, **final_vars_kwargs})
     
    else:
        # Save static parameters :
        for key, value in final_constant.items():
            trial.set_user_attr(key, value)

        final_vars_kwargs = dict()
        for parameter in final_vars:
            final_vars_kwargs[parameter.name] = parameter.apply(trial)
        
        return model(**{**final_constant, **final_vars_kwargs})


def get_opti_from_model(model, trial):
    data_dict = {}
    opti_dict = {}

    if not model.__dict__.get('opti_params', None) is None:
        if not model.opti_params.get("data", None) is None:
            for key, value in model.opti_params["data"].items():
                data_dict[key] = value

        if not model.opti_params.get("optim", None) is None:
            for key, value in model.opti_params["optim"].items():
                opti_dict[key] = value        

    if not model.__dict__.get("var_opti_params", None) is None:

        if not model.var_opti_params.get("data", None) is None:
            for parameter in model.var_opti_params["data"]:
                data_dict[parameter.name] = parameter.apply(trial=trial)
            
        if not model.var_opti_params.get("optim", None) is None:
            for parameter in model.var_opti_params["optim"]:
                opti_dict[parameter.name] = parameter.apply(trial=trial)

    return data_dict, opti_dict


if __name__ == "__main__":
    #def obj(trial):
    #    instanciate_model(BinMNIST, trial=trial)
    #    return randint(0,100)
#
    #study = optuna.create_study(study_name="lol")
    #study.optimize(obj, n_trials=3)
    instanciate_model(BinMNIST, trial=None, constants={'coco':111})
