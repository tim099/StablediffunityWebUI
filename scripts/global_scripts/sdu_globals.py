
def get_arg_val(args: dict, key: str, default_val):
    try:
        val = default_val
        if(key in args):
            if(type(val) is bool):
                val = args[key] == 'True'
            else:
                val = args[key]
        else:
            print("get_arg_val key:"+key+",not exist!!")
    except BaseException:
        print(f"get_arg_val key:"+key+",BaseException:{BaseException}")
    finally:
        return val;

class GlobalSetting():
    def __init__(self) -> None:
        self.args: dict = None
        self.FolderPath = 'None'
        self.OutputTensors = False
        self.LoadTensor = False
        self.LoadTensorFileName = 'None'
    def set_args(self, args: dict):
        self.args = args
        self.FolderPath = get_arg_val(args,'FolderPath', self.FolderPath)
        self.OutputTensors = get_arg_val(args,'OutputTensors', self.OutputTensors)
        self.LoadTensor = get_arg_val(args,'LoadTensor', self.LoadTensor)
        self.LoadTensorFileName = get_arg_val(args,'LoadTensorFileName', self.LoadTensorFileName)


    def info_str(self):
        if(self.args is None):
            return "None"
        import json
        return json.dumps(self.args)

global global_setting
global_setting: GlobalSetting = GlobalSetting()



