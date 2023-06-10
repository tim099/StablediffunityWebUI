import os
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException

import gradio as gr
import modules.launch_utils as launch_utils

from modules.api.models import *
from modules.api import api
from pathlib import Path

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

StablediffunityVersion = Path(os.path.join(Path(__location__).parent.absolute(),'Version.txt')).read_text()

def stablediffunity_api(_: gr.Blocks, app: FastAPI):
    @app.get("/stablediffunity/version")
    async def version():
        print("stablediffunity/version:" + StablediffunityVersion)
        return {"version": StablediffunityVersion}
    @app.post("/stablediffunity/git_clone")
    async def detect(
        url: str = Body("none", title='Url'),
        target_dir: str = Body("none", title='Dir'),
        branch: str = Body("none", title='Branch'),
    ):
        launch_utils.git_clone(url,target_dir,branch)
        clone_result = "git_clone url:" + url+",dir:"+target_dir+",branch:"+branch;
        print(clone_result)
        return {"git_clone": clone_result}


try:
    import modules.script_callbacks as script_callbacks
    print("[StableDiffUnity]  script_callbacks.on_app_started(stablediffunity_api) Version:" + StablediffunityVersion)
    script_callbacks.on_app_started(stablediffunity_api)
except:
    pass
