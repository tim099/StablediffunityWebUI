import os
from copy import copy
from enum import Enum
from pickle import TRUE
from typing import Tuple, List

from modules import img2img, script_callbacks
from scripts import external_code
#from scripts.global_scripts.sdu_globals import GlobalSetting
from scripts.global_scripts.sdu_globals import global_setting
class HijackData:

    def __init__(self, module, name, new_function):
        self.module = module
        self.name = name
        self.backup_func_name = "__original__" + name
        self.new_function = new_function

    def do_hijack(self):
        print("do_hijack name:"+self.name)
        #backup original function to backup_func_name attr
        if hasattr(self.module, self.name):
            setattr(self.module, self.backup_func_name, getattr(self.module, self.name))
        #hijack new_function to replace original function
        setattr(self.module, self.name, self.new_function)

    def undo_hijack(self):
        print("undo_hijack name:"+self.name)
        #check backup_func_name attr exist
        if hasattr(self.module, self.backup_func_name):
            #set original function back
            setattr(self.module, self.name, getattr(self.module, self.backup_func_name))
            #delete backup_func_name attr
            delattr(self.module, self.backup_func_name)

class Hijack:
    def __init__(self):
        self.hijack_list = []

    def do_hijack(self):
        print("SDU_do_hijack")
        script_callbacks.on_script_unloaded(self.undo_hijack)
        import k_diffusion.sampling
        self.add_hijack_data(k_diffusion.sampling,'sample_dpmpp_2m',sample_dpmpp_2m);

    def add_hijack_data(self, module, name, new_value):
        hijack_data = HijackData(module,name,new_value);
        hijack_data.do_hijack();
        self.hijack_list.append(hijack_data)
    def undo_hijack(self):
        print("SDU_undo_hijack")
        for hijack_data in self.hijack_list:
            hijack_data.undo_hijack()

        self.hijack_list.clear()


import torch
from tqdm.auto import trange, tqdm
@torch.no_grad()
def sample_dpmpp_2m(model, x, sigmas, extra_args=None, callback=None, disable=None):
    """DPM-Solver++(2M)."""
    extra_args = {} if extra_args is None else extra_args
    s_in = x.new_ones([x.shape[0]])
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    old_denoised = None


    #from sdu_globals import global_setting
    from pathlib import Path
    from datetime import datetime

    
    current_time = datetime.now().strftime("%H_%M_%S")
    print("SDU global_setting.OutputPath:" + global_setting.OutputPath)
    print("SDU global_setting.OutputTensors:" + str(global_setting.OutputTensors)+",type:"+type(global_setting.OutputTensors).__name__)
    #print("SDU global_setting.info_str:" + GlobalSetting.info_str())
    print("SDU_s_in:" + str(s_in.item()))
    #torch.log(sigmas)
    print("SDU_sigmas:" + ", ".join(f'{x.item():.3f}' for x in sigmas)+"\n",flush=True)

    output_path = Path(global_setting.OutputPath, "tensors")
    if global_setting.OutputTensors == True:
        print("SDU OutputTensors!!")
        if not os.path.exists(output_path):
            # Create a new directory because it does not exist
            os.makedirs(output_path)
    else:
        print("SDU Dont OutputTensors!!")

    for i in trange(len(sigmas) - 1, disable=disable):

        denoised = model(x, sigmas[i] * s_in, **extra_args)

        if callback is not None:
            callback({'x': x, 'i': i, 'sigma': sigmas[i], 'sigma_hat': sigmas[i], 'denoised': denoised})
        t, t_next = t_fn(sigmas[i]), t_fn(sigmas[i + 1])
        h = t_next - t
        if old_denoised is None or sigmas[i + 1] == 0:
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised
        else:
            h_last = t - t_fn(sigmas[i - 1])
            r = h_last / h
            denoised_d = (1 + 1 / (2 * r)) * denoised - (1 / (2 * r)) * old_denoised
            x = (sigma_fn(t_next) / sigma_fn(t)) * x - (-h).expm1() * denoised_d
        old_denoised = denoised
        if global_setting.OutputTensors == True:
            x_path = Path(output_path,"x_"+current_time+"__"+ str(i)+".pt")
            #print("x_path:"+str(x_path))
            torch.save(x, x_path)
    return x


instance = Hijack()
