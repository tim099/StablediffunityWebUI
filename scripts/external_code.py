from modules.api import api
from typing import List, Any, Optional, Union, Tuple, Dict
from modules import scripts, processing, shared

import modules.scripts as scripts

class StableDiffUnityUnit:
    """
    Represents an entire StableDiffUnity processing unit.
    """

    def __init__(
        self,
        enabled: bool=True,
        **_kwargs,
    ):
        self.enabled = enabled

    def __eq__(self, other):
        if not isinstance(other, StableDiffUnityUnit):
            return False

        return vars(self) == vars(other)


def get_all_units_in_processing(p: processing.StableDiffusionProcessing) -> List[StableDiffUnityUnit]:
    """
    Fetch StableDiffUnityUnit processing units from a StableDiffusionProcessing.
    """

    return get_all_units(p.scripts, p.script_args)


def get_all_units(script_runner: scripts.ScriptRunner, script_args: List[Any]) -> List[StableDiffUnityUnit]:
    """
    Fetch StableDiffUnityUnit processing units from an existing script runner.
    Use this function to fetch units from the list of all scripts arguments.
    """

    sdu_script = find_sdu_script(script_runner)
    if sdu_script:
        return get_all_units_from(script_args[sdu_script.args_from:sdu_script.args_to])

    return []


def get_all_units_from(script_args: List[Any]) -> List[StableDiffUnityUnit]:
    """
    Fetch StableDiffUnityUnit processing units from StableDiffUnityUnit script arguments.
    Use `external_code.get_all_units` to fetch units from the list of all scripts arguments.
    """

    units = []
    i = 0
    while i < len(script_args):
        if script_args[i] is not None:
            units.append(to_processing_unit(script_args[i]))
        i += 1

    return units


def get_single_unit_from(script_args: List[Any], index: int=0) -> Optional[StableDiffUnityUnit]:
    """
    Fetch a single StableDiffUnityUnit processing unit from StableDiffUnityUnit script arguments.
    The list must not contain script positional arguments. It must only contain processing units.
    """

    i = 0
    while i < len(script_args) and index >= 0:
        if index == 0 and script_args[i] is not None:
            return to_processing_unit(script_args[i])
        i += 1

        index -= 1

    return None

def to_processing_unit(unit: Union[Dict[str, Any], StableDiffUnityUnit]) -> StableDiffUnityUnit:

    return unit


def update_sdu_script_in_processing(
    p: processing.StableDiffusionProcessing,
    sdu_units: List[StableDiffUnityUnit],
    **_kwargs, # for backwards compatibility
):
    """
    Update the arguments of the StableDiffUnityUnit script in `p.script_args` in place, reading from `sdu_units`.
    `sdu_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.

    Does not update `p.script_args` if any of the folling is true:
    - ControlNet is not present in `p.scripts`
    - `p.script_args` is not filled with script arguments for scripts that are processed before ControlNet
    """

    cn_units_type = type(sdu_units) if type(sdu_units) in (list, tuple) else list
    script_args = list(p.script_args)
    update_sdu_script_in_place(p.scripts, script_args, sdu_units)
    p.script_args = cn_units_type(script_args)


def update_sdu_script_in_place(
    script_runner: scripts.ScriptRunner,
    script_args: List[Any],
    sdu_units: List[StableDiffUnityUnit],
    **_kwargs, # for backwards compatibility
):
    """
    Update the arguments of the StableDiffUnityUnit script in `script_args` in place, reading from `sdu_units`.
    `sdu_units` and its elements are not modified. You can call this function repeatedly, as many times as you want.
    """

    sdu_script = find_sdu_script(script_runner)
    if sdu_script is None or len(script_args) < sdu_script.args_from:
        return

    # fill in remaining parameters to satisfy max models, just in case script needs it.
    max_models = 3
    sdu_units = sdu_units + [StableDiffUnityUnit(enabled=False)] * max(max_models - len(sdu_units), 0)

    script_args_diff = 0
    for script in script_runner.alwayson_scripts:
        if script is sdu_script:
            script_args_diff = len(sdu_units) - (sdu_script.args_to - sdu_script.args_from)
            script_args[script.args_from:script.args_to] = sdu_units
            script.args_to = script.args_from + len(sdu_units)
        else:
            script.args_from += script_args_diff
            script.args_to += script_args_diff



def find_sdu_script(script_runner: scripts.ScriptRunner) -> Optional[scripts.Script]:
    """
    Find the StableDiffUnity script in `script_runner`. Returns `None` if `script_runner` does not contain a StableDiffUnity script.
    """

    if script_runner is None:
        return None
    from scripts.stablediffunity import SDU_Title
    for script in script_runner.alwayson_scripts:
        if script.title() == SDU_Title:
            return script

