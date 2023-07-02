#from scripts.global_scripts.sdu_globals import sdu_globals_util as util
#import sdu_globals_util as util
from math import fabs
import os
from scripts.global_scripts.sdu_globals_util import get_arg_val
import torch
from pathlib import Path
from datetime import datetime
from scripts.global_scripts.sdu_sample_data import SampleData

class SDU_WebUICMD():
    def __init__(self) -> None:
        pass
    def set_args(self, args: dict):
        print("SDU_WebUICMD set_args")
    def sample_start(self):
        pass
    
    def skip_sample(self, data:SampleData) -> bool: #skip current sample step if return true
        return False
    def trigger(self, data:SampleData):
        pass
        #print("SDU_WebUICMD trigger", flush = True)
    def sample_end(self, data: SampleData):
        print(f"SDU_WebUICMD sample_end step:{str(data.step)}", flush = True)
        self.trigger(data)

class SDU_WebUICMDOutputTensors(SDU_WebUICMD):
    def __init__(self) -> None:
        self.FolderPath = 'None'
        self.OutputAtSteps:list[int] = []
        self.CurrentTime = 'None'
        self.OutputJsonTensor = False
    def set_args(self, args: dict):
        print("SDU_WebUICMDOutputTensors set_args")

        self.FolderPath = get_arg_val(args,'FolderPath', self.FolderPath)

        self.OutputAtSteps.clear()
        self.OutputAtSteps = get_arg_val(args,'OutputAtSteps', self.OutputAtSteps)
        self.OutputJsonTensor = get_arg_val(args,'OutputJsonTensor', self.OutputJsonTensor)
        print("OutputAtSteps:"+", ".join(f'{"{:02d}".format(x)}' for x in self.OutputAtSteps))
    def sample_start(self):
        #print("SDU_WebUICMDOutputTensors sample_start")
        self.CurrentTime = datetime.now().strftime("%H_%M_%S")
        self.folder_path = Path(self.FolderPath)#, "tensors"
        if not os.path.exists(self.folder_path):
            # Create a new directory because it does not exist
            os.makedirs(self.folder_path)

        pass
    def trigger(self, data:SampleData):
        if data.step in self.OutputAtSteps:
            from scripts.global_scripts.sdu_globals import global_setting
            #x_path = Path(self.folder_path,"x_"+self.CurrentTime+"__"+ "{:03d}".format(data.step)+".pt")
            file_name = global_setting.CurOutputImageName+"__"+ "{:03d}".format(data.step)
            x_path = Path(self.folder_path,file_name+".pt")
            #print("x_path:"+str(x_path))
            torch.save(data.x, x_path)
            if self.OutputJsonTensor:
                import json
                data_list = data.x.tolist()
                with open(Path(self.folder_path,file_name+".json"), 'w') as f:
                    json.dump(data_list, f)
            #for size in data.x.shape:
            #    pass
            #if self.OutputJsonTensor:
            #    for x in data.x:
            #        for y in x: 
            #            for z in y:
            #                for w in z:
            #                    json
        #print(f"SDU_WebUICMDOutputTensors trigger step:{str(step)}")

class SDU_WebUICMDLoadTensor(SDU_WebUICMD):
    def __init__(self) -> None:
        self.FolderPath = 'None'
        self.LoadAtStep = 0
        self.LoadTensorFileName = 'None'
    def set_args(self, args: dict):
        print("SDU_WebUICMDLoadTensor set_args")

        self.FolderPath = get_arg_val(args,'FolderPath', self.FolderPath)
        self.LoadAtStep = get_arg_val(args,'LoadAtStep', self.LoadAtStep)
        self.LoadTensorFileName = get_arg_val(args,'LoadTensorFileName', self.LoadTensorFileName)
    def sample_start(self):
        #print("SDU_WebUICMDLoadTensor sample_start")
        pass

    #skip current sample if return true
    def skip_sample(self, data:SampleData) -> bool:
        if(self.LoadAtStep < 0 or self.LoadAtStep > data.step):
            return True
        return False
    def trigger(self, data:SampleData):
        if(self.LoadAtStep == data.step or (self.LoadAtStep < 0 and data.step == 0)):
            tensor_path = Path(self.FolderPath, self.LoadTensorFileName)
            if os.path.exists(tensor_path):
                print("LoadTensor tensor_path:"+str(tensor_path)+",step:"+str(data.step))
                data.x = torch.load(tensor_path)
            else:
                print("!os.path.exists(tensor_path) tensor_path:"+str(tensor_path))
