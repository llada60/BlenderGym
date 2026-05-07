import bpy
import random
import json
import os
import sys
import site
import importlib.util
from sys import platform

def _ensure_gin_importable():
    try:
        import gin  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    user_site = site.getusersitepackages()
    if not user_site:
        return

    gin_init = os.path.join(user_site, "gin", "__init__.py")
    gin_pkg_dir = os.path.dirname(gin_init)
    if not os.path.isfile(gin_init):
        return

    spec = importlib.util.spec_from_file_location(
        "gin",
        gin_init,
        submodule_search_locations=[gin_pkg_dir],
    )
    if spec is None or spec.loader is None:
        return

    gin_module = importlib.util.module_from_spec(spec)
    sys.modules["gin"] = gin_module
    spec.loader.exec_module(gin_module)


if __name__ == "__main__":
    _ensure_gin_importable()

    code_fpath = sys.argv[6]  # Path to the code file
    rendering_dir = sys.argv[7] # Path to save the rendering from camera1

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

    # Set max samples to 1024
    bpy.context.scene.cycles.samples = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Read and execute the code from the specified file
    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except:
        raise ValueError

    # Render from camera1
    if 'Camera' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render.png')
        bpy.ops.render.render(write_still=True)

    # Render from camera1
    if 'Camera1' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera1']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render1.png')
        bpy.ops.render.render(write_still=True)

    # Render from camera2
    if 'Camera2' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera2']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render2.png')
        bpy.ops.render.render(write_still=True)

