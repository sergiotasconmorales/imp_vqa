import yaml
import os

def read_config(path_config, return_config_name=False):
    """Function to read the config file from path_config

    Parameters
    ----------
    path_config : str
        path to config file

    Returns
    -------
    dict
        parsed config file
    """
    with open(path_config, "r") as ymlfile:
        cfg = yaml.load(ymlfile, Loader=yaml.FullLoader)
    if return_config_name:
        cfg_name_ext = os.path.split(path_config)[-1]
        cfg_name = cfg_name_ext.split('.')[0]
        return cfg, cfg_name
    else:
        return cfg

def update_args(args, updated_values, exp_name):
    # update path to experiment logs first
    updated_values['output'] = os.path.join(updated_values['output'], exp_name)
    os.makedirs(updated_values['output'], exist_ok=True)
    # Updates args according to info from config file, which is contained in dict updated_values
    args_dict = vars(args)
    for k,v in updated_values.items():
        setattr(args, k, v)

    return args