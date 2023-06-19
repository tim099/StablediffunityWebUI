import gc
import os
from collections import OrderedDict
from copy import copy
from typing import Dict, Optional, Tuple
import importlib
import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images

from einops import rearrange
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.images import save_image

import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageFilter, ImageOps


SDU_Title = "StableDiffUnity"
class Script(scripts.Script):

    def __init__(self) -> None:
        super().__init__()
        print("StableDiffUnity Script __init__()")

    def title(self):
        return SDU_Title

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        sd_model = p.sd_model
        print("StableDiffUnity Script process(self, p, *args),p.width:"+str(p.width)+",p.height:"+str(p.height))

    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs.get('images', [])
        return

    def postprocess(self, p, processed, *args):
        self.post_processors = []
        gc.collect()
        devices.torch_gc()


def find_sdu_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the StableDiffUnity script in `script_runner`. Returns `None` if `script_runner` does not contain a StableDiffUnity script.
    """

    if script_runner is None:
        return None

    for script in script_runner.alwayson_scripts:
        if script.title() == SDU_Title:
            return script