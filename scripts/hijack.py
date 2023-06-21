import os
from copy import copy
from enum import Enum
from typing import Tuple, List

from modules import img2img, script_callbacks
from scripts import external_code
import k_diffusion.sampling as sampling
OriginalFuncName = 'sample_dpmpp_2m'
NewFuncName = '__original_sample_dpmpp_2m';
target_module = sampling


class HijackData:
    def __init__(self, module, name, new_name, new_value):
        self.module = module
        self.name = name
        self.new_name = new_name
        self.new_value = new_value

    def do_hijack(self):
        print("do_hijack name:"+self.name)
        setattr(self.module, self.new_name, getattr(self.module, self.name))
        setattr(self.module, self.name, self.new_value)

    def undo_hijack(self):
        print("undo_hijack name:"+self.name)
        if hasattr(self.module, self.new_name):
            setattr(self.module, self.name, getattr(self.module, self.new_name))
            delattr(self.module, self.new_name)

class Hijack:
    def __init__(self):
        self.postprocess_batch_callbacks = [self.on_postprocess_batch]
        self.hijack_list = []
    def process_batch_hijack(self, p, *args, **kwargs):
        print("SDU_process_batch_hijack")
        try:
            return getattr(target_module, NewFuncName)(p, *args, **kwargs)
        finally:
            self.dispatch_callbacks(self.postprocess_batch_callbacks, p)

    def on_postprocess_batch(self, p, *args):
        print("on_postprocess_batch")

    def do_hijack(self):
        print("SDU_do_hijack")
        script_callbacks.on_script_unloaded(self.undo_hijack)

        hijack_data = HijackData(target_module,OriginalFuncName,NewFuncName,sample_dpmpp_2m);
        hijack_data.do_hijack();
        self.hijack_list.append(hijack_data)

    def undo_hijack(self):
        print("SDU_undo_hijack")
        for hijack_data in self.hijack_list:
            hijack_data.undo_hijack()

        self.hijack_list.clear()



    def dispatch_callbacks(self, callbacks, *args):
        for callback in callbacks:
            callback(*args)

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

    from pathlib import Path
    from datetime import datetime
    from modules.processing import folder_path
    current_time = datetime.now().strftime("%H_%M_%S")
    print("SDU_s_in:" + str(s_in.item()))
    torch.log(sigmas)
    print("SDU_sigmas:" + ", ".join(f'{x.item():.3f}' for x in sigmas)+"\n",flush=True)
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
        #x_path = Path(folder_path,"x_"+current_time+"__"+ str(i)+".pt")
        #torch.save(x, x_path)
    return x


instance = Hijack()
