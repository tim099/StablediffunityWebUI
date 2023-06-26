


class GlobalSetting():
    global OutputPath
    OutputPath = 'None'
    def __init__(self) -> None:
        self.args: dict = None
        self.stablediffunity = 'Default'
    def set_args(self, args: dict):
        self.args = args
        self.stablediffunity = args['stablediffunity']
        GlobalSetting.OutputPath = args['OutputPath']
        print("GlobalSetting.OutputPath:"+GlobalSetting.OutputPath)
    def info_str(self):
        if(self.args is None):
            return "None"
        import json
        return json.dumps(self.args)

global global_setting
global_setting: GlobalSetting = GlobalSetting()


