import gradio as gr
from scripts import (
    external_code,
)


class UIStableDiffUnityUnit(external_code.StableDiffUnityUnit):
    """The data class that stores all states of a StableDiffUnityUnit."""

    def __init__(
        self,
        enabled: bool = True,
        *args,
        **kwargs,
    ):
        super().__init__(enabled, *args, **kwargs)

class StableDiffUnityUIGroup(object):
    def __init__(
        self,
    ):
        self.default_unit = UIStableDiffUnityUnit()

    def render_and_register_unit(self, tabname: str, is_img2img: bool):
        print("SDU render_and_register_unit tabname:"+tabname)

        unit = gr.State(self.default_unit)
        return unit

