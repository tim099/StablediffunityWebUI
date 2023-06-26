
class GlobalSetting():
    #global OutputPath
    #OutputPath = 'None'
    def __init__(self) -> None:
        self.args: dict = None
        self.FolderPath = 'None'
        self.OutputTensors = False
        self.LoadTensor = False
        self.LoadTensorFileName = 'None'
    def set_args(self, args: dict):
        self.args = args
        #GlobalSetting.OutputPath = args['OutputPath']
        #print("GlobalSetting.OutputPath:"+GlobalSetting.OutputPath)
        self.FolderPath = args['FolderPath']
        self.OutputTensors = (args['OutputTensors'] == 'True')
        self.LoadTensor = (args['LoadTensor'] == 'True')
        self.LoadTensorFileName = args['LoadTensorFileName']

    def info_str(self):
        if(self.args is None):
            return "None"
        import json
        return json.dumps(self.args)

global global_setting
global_setting: GlobalSetting = GlobalSetting()



