import torch
import json
import numpy as np

def get_activation(target_activ):
    activ_all = {   'swish'     : torch.nn.SiLU(),
                    'elu'       : torch.nn.ELU(),
                    'linear'    : torch.nn.Identity(),
                    'relu'      : torch.nn.ReLU(),
                    'sigmoid'   : torch.nn.Sigmoid(),
                    'tanh'      : torch.nn.Tanh(),
                    'softmin'   : torch.nn.Softmin(),
                    'softmax'   : torch.nn.Softmax(),
                }
    assert (target_activ in activ_all.keys()), '{}: unsupported activation specified'.format(target_activ)
    return activ_all[target_activ]

def get_device():
    device = (
                "cuda"
                if torch.cuda.is_available()
                else "mps"
                if torch.backends.mps.is_available()
                else "cpu"
                )   
    print(f"Using {device} device") 

    return device

# NpEncoder taken from:
# https://stackoverflow.com/questions/50916422/python-typeerror-object-of-type-int64-is-not-json-serializable
class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NpEncoder, self).default(obj)