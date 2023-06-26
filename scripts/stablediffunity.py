import gc
import os
from collections import OrderedDict
from copy import copy
from typing import Dict, Optional, Tuple
import importlib
import modules.scripts as webui_scripts
from modules import shared, devices, script_callbacks, processing, masking, images

from einops import rearrange
from modules.processing import StableDiffusionProcessingImg2Img, StableDiffusionProcessingTxt2Img
from modules.images import save_image
from scripts.ui.ui_group import StableDiffUnityUIGroup, UIStableDiffUnityUnit
import cv2
import numpy as np
import torch
import gradio as gr
from pathlib import Path
from PIL import Image, ImageFilter, ImageOps
#from scripts.global_scripts.sdu_globals import GlobalSetting
from scripts.global_scripts.sdu_globals import global_setting

SDU_Title = "StableDiffUnity"
__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

StablediffunityVersion = Path(os.path.join(Path(__location__).parent.absolute(),'Version.txt')).read_text()



class Script(webui_scripts.Script):

    def __init__(self) -> None:
        super().__init__()
        print("StableDiffUnity Script __init__()")
    def title(self):
        return SDU_Title

    def show(self, is_img2img):
        return webui_scripts.AlwaysVisible
    
    def process(self, p, *args):
        """
        This function is called before processing begins for AlwaysVisible scripts.
        You can modify the processing object (p) here, inject hooks, etc.
        args contains all values returned by components from ui()
        """
        sd_model = p.sd_model
        #from sdu_globals import sdu_globals

        global_setting.set_args(p.script_args[self.args_from])
        print("StableDiffUnity Script process(self, p, *args),p.width:"+str(p.width)+",p.height:"+str(p.height))
        print("StableDiffUnity ,args_from:"+str(self.args_from)+",args_to:"+str(self.args_to))
        print("StableDiffUnity ,global_setting:"+global_setting.info_str())
        print("StableDiffUnity ,GlobalSetting.OutputPath:"+global_setting.FolderPath)
    def uigroup(self, tabname: str, is_img2img: bool, elem_id_tabname: str):
        group = StableDiffUnityUIGroup()
        return group.render_and_register_unit(tabname, is_img2img)
    def ui(self, is_img2img):
        """this function should create gradio UI elements. See https://gradio.app/docs/#components
        The return value should be an array of all components that are used in processing.
        Values of those returned components will be passed to run() and process() functions.
        """
        self.infotext_fields = []
        self.paste_field_names = []
        controls = ()
        
        elem_id_tabname = ("img2img" if is_img2img else "txt2img") + "_stablediffunity"
        max_models = 1
        with gr.Group(elem_id=elem_id_tabname):
            with gr.Accordion(f"StableDiffUnity {StablediffunityVersion}", open = False, elem_id="stablediffunity"):
                with gr.Tabs(elem_id=f"{elem_id_tabname}_tabs"):
                    for i in range(max_models):
                        with gr.Tab(f"StableDiffUnity Unit {i}", elem_classes=['sdu-unit-tab']):
                            controls += (self.uigroup(f"StableDiffUnity-{i}", is_img2img, elem_id_tabname),)


        if shared.opts.data.get("stablediffunity_sync_field_args", False):
            for _, field_name in self.infotext_fields:
                self.paste_field_names.append(field_name)

        return controls
    def postprocess_batch(self, p, *args, **kwargs):
        images = kwargs.get('images', [])
        return

    def postprocess(self, p, processed, *args):
        self.post_processors = []
        gc.collect()
        devices.torch_gc()

from scripts import hijack

hijack.instance.do_hijack()