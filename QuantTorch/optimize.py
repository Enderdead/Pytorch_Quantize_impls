import argparse
import importlib.util
import os.path
import configparser
import optuna
from threading import Thread
from train.instantiate import *
from train.train import train
from multiprocessing import Process
from train.metrics import Accuracy
from dataset.mnist import get_mnist
import torch

1/0

parser = argparse.ArgumentParser(description='Quant tolls Optimizer')


parser.add_argument('-j','--jobs', type=int, default=4, metavar='j',
                    help='Threads to launch for this training')

parser.add_argument('-n','--n_tials', type=int, default=7, metavar='n',
                    help='Number of sampling to compute')

parser.add_argument('--gpus', type=str, default="",
                    help='gpus used for training - e.g 0,1,3 or 2')

parser.add_argument("-c", "--config_file", type=str, default="./config/optimize.conf", )                    

parser.add_argument("study_name", type=str, help="Study name to use for this optim")
parser.add_argument("model", type=str, help="Path to the model")

args = parser.parse_args()

#TODO look if the model name is on template folder.

# Load model 
spec = importlib.util.spec_from_file_location(os.path.split(args.model)[-1], args.model)
model = importlib.util.module_from_spec(spec)
spec.loader.exec_module(model)

# Load config parser
#config = configparser.ConfigParser()
#config.read(args.config_file)



# setup work to do per thread
worker = [ {"device":None, "n_trials": args.n_tials//args.jobs} for _ in range(args.jobs)]

for index in range(args.n_tials%args.jobs):
    worker[index]["n_trials"] +=1

# Setup device 
if args.gpus =="":
    # If no gpu setup only use cpu
    for work in worker:
        work["device"] = "cpu"
else:
    gpus = args.gpus.split(",")
    for index, work in enumerate(worker):
        work["device"] = "cuda:{}".format(gpus[index%(len(gpus))])


def thread_main(device, nb_trial, job_index):
    os.execlp("python3","python3","./utils/jobs/optimize.py", str(device), str(nb_trial), str(job_index), args.model, args.study_name)


jobs = [ Process(target=thread_main, args=(worker[index]["device"],worker[index]["n_trials"], index)) for index in range(args.jobs)]

for job in jobs:
    job.start()

for job in jobs:
    job.join()
"""

def thread_main(device, nb_trial, job_index):

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
        
        print(scheduled_opti_conf)
        print(device)
        train(net, train_set, valid_set, device=device, save_path="/tmp/Optimize/job_{}".format(job_index), early_stopping=False,
                    opti=optimizer, loss=torch.nn.CrossEntropyLoss(), max_epoch=max_epoch,
                    static_opti_conf=None, scheduled_opti_conf=scheduled_opti_conf, accuracy_method=Accuracy(10))
        

        return 0

    study = optuna.create_study(study_name=args.study_name, load_if_exists=True)
    study.optimize(objectif, n_trials=nb_trial)




jobs = [ Process(target=thread_main, args=(worker[index]["device"],worker[index]["n_trials"], index)) for index in range(args.jobs)]


for job in jobs:
    job.start()

for job in jobs:
    job.join()

"""