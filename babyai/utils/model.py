import os
import torch

from .. import utils


def get_model_dir(model_name):
    return os.path.join(utils.storage_dir(), "models", model_name)


def get_model_path(model_name):
    return os.path.join(get_model_dir(model_name), "model.pt")

def get_optimizer_path(model_name):
    return os.path.join(get_model_dir(model_name), "optimizer.pt")

def load_model(model_name, raise_not_found=True):
    path = get_model_path(model_name)
    opt_path = get_optimizer_path(model_name)
    try:
        if torch.cuda.is_available():
            model = torch.load(path)
        else:
            model = torch.load(path, map_location=torch.device("cpu"))
        model.eval()
        try:
            opt_state = torch.load(opt_path)['optimizer_state_dict']
            schd_state = torch.load(opt_path)['scheduler_state_dict']
        except FileNotFoundError:
            opt_state = None
            schd_state = None
        return {
            'model': model,
            'optimizer': opt_state,
            'scheduler': schd_state
        }
    except FileNotFoundError:
        if raise_not_found:
            raise FileNotFoundError("No model found at {}".format(path))


def save_model(model, model_name):
    path = get_model_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save(model, path)

def save_optimizer(optimizer, scheduler, model_name):
    path = get_optimizer_path(model_name)
    utils.create_folders_if_necessary(path)
    torch.save({
                    'optimizer_state_dict': optimizer.state_dict(),
                    'scheduler_state_dict': scheduler.state_dict()
                }, path)
