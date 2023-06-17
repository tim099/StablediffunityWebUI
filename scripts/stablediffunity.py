import gc
import os
from collections import OrderedDict
from copy import copy
from typing import Dict, Optional, Tuple
import importlib
import modules.scripts as scripts
from modules import shared, devices, script_callbacks, processing, masking, images
import gradio as gr

from einops import rearrange
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.images import save_image

import cv2
import numpy as np
import torch

from pathlib import Path
from PIL import Image, ImageFilter, ImageOps

gradio_compat = True
try:
    from distutils.version import LooseVersion
    from importlib_metadata import version
    if LooseVersion(version("gradio")) < LooseVersion("3.10"):
        gradio_compat = False
except ImportError:
    pass


# Gradio 3.32 bug fix
import tempfile
gradio_tempfile_path = os.path.join(tempfile.gettempdir(), 'gradio')
os.makedirs(gradio_tempfile_path, exist_ok=True)



class Script(scripts.Script):

    def __init__(self) -> None:
        super().__init__()


    def title(self):
        return "StableDiffUnity"

    def show(self, is_img2img):
        return scripts.AlwaysVisible
    
    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """

        sd_model = p.sd_model


    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs.get('images', [])
        return

    def postprocess(self, p, processed, *args):
        self.post_processors = []
        gc.collect()
        devices.torch_gc()


