import torch
class SampleData():
    def __init__(self) -> None:
        self.x:torch.Tensor
        self.step:int = 0
        self.skip_steps:list[int] = []


