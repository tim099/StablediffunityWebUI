import os
import numpy as np
from fastapi import FastAPI, Body
from fastapi.exceptions import HTTPException

import gradio as gr

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
    


try:
    import modules.script_callbacks as script_callbacks
    print("[StableDiffUnity]  script_callbacks.on_app_started(stablediffunity_api) Version:" + StablediffunityVersion)
    script_callbacks.on_app_started(stablediffunity_api)
except:
    pass
