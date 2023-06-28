import os
from copy import copy
from enum import Enum
from pickle import TRUE
from typing import Tuple, List

from modules import img2img, script_callbacks
from scripts import external_code

from scripts.global_scripts.sdu_sample_data import SampleData
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
        import scripts.sdu_sampling
        from scripts.sdu_sampling import sample_dpmpp_2m
        #self.add_hijack_data(k_diffusion.sampling,'sample_dpmpp_2m',scripts.sdu_sampling.sample_dpmpp_2m);
        #self.add_hijack_data(k_diffusion.sampling,'sample_euler',scripts.sdu_sampling.sample_euler);
        self.add_hijack_sampling(k_diffusion.sampling, scripts.sdu_sampling, 'sample_dpmpp_2m')
        self.add_hijack_sampling(k_diffusion.sampling, scripts.sdu_sampling, 'sample_euler')
        self.add_hijack_sampling(k_diffusion.sampling, scripts.sdu_sampling, 'sample_euler_ancestral')
        self.add_hijack_sampling(k_diffusion.sampling, scripts.sdu_sampling, 'sample_lms')
        self.add_hijack_sampling(k_diffusion.sampling, scripts.sdu_sampling, 'sample_heun')
        self.add_hijack_sampling(k_diffusion.sampling, scripts.sdu_sampling, 'sample_dpm_2')
        self.add_hijack_sampling(k_diffusion.sampling, scripts.sdu_sampling, 'sample_dpm_2_ancestral')

    def add_hijack_sampling(self, module, new_module, name):
        self.add_hijack_data(module, name, getattr(new_module,name));

    def add_hijack_data(self, module, name, new_value):
        hijack_data = HijackData(module,name,new_value);
        hijack_data.do_hijack();
        self.hijack_list.append(hijack_data)
    def undo_hijack(self):
        print("SDU_undo_hijack")
        for hijack_data in self.hijack_list:
            hijack_data.undo_hijack()

        self.hijack_list.clear()


instance = Hijack()
