import os
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException
from pathlib import Path

import gradio as gr
import modules.launch_utils as launch_utils
import modules.sd_vae as sd_vae
from modules.api.models import *
from modules.api import api

__location__ = os.path.realpath(os.path.join(os.getcwd(), os.path.dirname(__file__)))

StablediffunityVersion = Path(os.path.join(Path(__location__).parent.absolute(),'Version.txt')).read_text()

def stablediffunity_api(_: gr.Blocks, app: FastAPI):
    @app.get("/stablediffunity/version")
    async def version():
        print("stablediffunity/version:" + StablediffunityVersion)
        return {"version": StablediffunityVersion}

    @app.get("/stablediffunity/sd-vae")
    async def vae():
        print("/stablediffunity/sd-vae")
        sd_vae.refresh_vae_list();
        return {"VAE": [{"name": x, "path": sd_vae.vae_dict[x]} for x in sd_vae.vae_dict.keys()]}

    @app.post("/stablediffunity/set-sd-vae")
    async def set_vae(
        vae: str = Body("none", title='Vae'),
        path: str = Body("none", title='Path'),
    ):
        sd_vae.reload_vae_weights(None, path)
        print("/stablediffunity/set-sd-va vae:"+vae+",path"+path)
        return {"VAE": vae}

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
