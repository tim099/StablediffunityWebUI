from scripts.global_scripts import sdu_webui_cmd as webui_cmd
#from scripts.global_scripts.sdu_globals import sdu_globals_util as util
from scripts.global_scripts.sdu_globals_util import get_arg_val
from scripts.global_scripts.sdu_sample_data import SampleData
import torch
import json


class GlobalSetting():
    def __init__(self) -> None:
        self.args: dict = None
        #self.FolderPath = 'None'
        #self.LoadTensor = False
        #self.LoadTensorFileName = 'None'
        self.WebUICMDs: list[webui_cmd.SDU_WebUICMD] = []
    def set_args(self, args: dict):
        self.args = args
        #self.FolderPath = get_arg_val(args,'FolderPath', self.FolderPath)
        #self.LoadTensor = get_arg_val(args,'LoadTensor', self.LoadTensor)
        #self.LoadTensorFileName = get_arg_val(args,'LoadTensorFileName', self.LoadTensorFileName)

        self.WebUICMDs.clear()
        cmds = get_arg_val(args,'WebUICMDs', [])
        print("cmds:"+json.dumps(cmds))
        for cmd in cmds:
            class_name = get_arg_val(cmd,'Class','None')
            print("class_name:"+class_name)
            class_data = get_arg_val(cmd,'Data',None)
            print("class_data:"+json.dumps(class_data))

            cls = getattr(webui_cmd, class_name) # get class
            cmd_obj: webui_cmd.SDU_WebUICMD = cls() # create object if class
            cmd_obj.set_args(class_data)
            self.WebUICMDs.append(cmd_obj)
    def info_str(self):
        if(self.args is None):
            return "None"
        return json.dumps(self.args)

    def sample_start(self):
        for cmd in self.WebUICMDs:
            cmd.sample_start()
    def skip_sample(self, data:SampleData) -> bool: #skip current sample step if return true
        for cmd in self.WebUICMDs:
            if cmd.skip_sample(data):
                data.skip_steps.append(data.step)
                return True
        return False
    def trigger(self, data: SampleData):
        for cmd in self.WebUICMDs:
            cmd.trigger(data)

global global_setting
global_setting: GlobalSetting = GlobalSetting()




