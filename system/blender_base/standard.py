"""
Given a python file (indicated inthe commandline path), render the material output.
"""

import bpy
import random
import json
import os
import sys
from sys import platform


# def get_material_from_code(code_fpath):
#     assert os.path.exists(code_fpath)
#     import pdb; pdb.set_trace()
#     with open(code_fpath, "r") as f:
#         code = f.read()
#     exec(code)
#     return material 


if __name__ == "__main__":

    code_fpath = sys.argv[6]  # TODO: allow a folder to be given, each with a possible guess.
    rendering_fpath = sys.argv[7] # rendering

    # Enable GPU rendering when the current Blender build exposes a supported device type.
    bpy.context.scene.render.engine = 'CYCLES'
    cycles_prefs = bpy.context.preferences.addons['cycles'].preferences

    preferred_device_types = []
    if platform == "darwin":
        preferred_device_types = ["METAL", "NONE"]
    elif platform == "linux" or platform == "linux2":
        preferred_device_types = ["OPTIX", "CUDA", "NONE"]
    elif platform == "win32":
        preferred_device_types = ["OPTIX", "CUDA", "NONE"]
    else:
        preferred_device_types = ["NONE"]

    selected_device_type = "NONE"
    for device_type in preferred_device_types:
        try:
            cycles_prefs.compute_device_type = device_type
            selected_device_type = device_type
            break
        except TypeError:
            continue

    cycles_prefs.get_devices()

    use_gpu = selected_device_type != "NONE"
    for device in cycles_prefs.devices:
        if use_gpu:
            device.use = device.type in {"GPU", "METAL", "OPTIX", "CUDA"}
        else:
            device.use = device.type == "CPU"

    bpy.context.scene.cycles.device = 'GPU' if use_gpu else 'CPU'

    # Setting up rendering resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Set max samples to 512
    bpy.context.scene.cycles.samples = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError
    
    # render, and save.
    bpy.context.scene.camera = bpy.data.objects['Camera1']
    bpy.context.scene.render.image_settings.file_format = 'PNG'
    bpy.context.scene.render.filepath = rendering_fpath
    bpy.ops.render.render(write_still=True)
