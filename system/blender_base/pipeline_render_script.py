import bpy
import random
import json
import os
import sys
import site
import importlib.util
from sys import platform


def _set_cycles_device(device_type):
    cycles_prefs = bpy.context.preferences.addons["cycles"].preferences
    cycles_prefs.compute_device_type = device_type
    cycles_prefs.get_devices()
    use_gpu = device_type != "NONE"
    for device in cycles_prefs.devices:
        device.use = use_gpu and device.type in {"GPU", "METAL", "OPTIX", "CUDA"}
    bpy.context.scene.cycles.device = "GPU" if use_gpu else "CPU"


def _configure_cycles_device():
    force_cpu = os.environ.get("BLENDERGYM_FORCE_CPU", "").lower() in {"1", "true", "yes"}
    if force_cpu:
        _set_cycles_device("NONE")
        return "NONE"

    if platform == "darwin":
        preferred_device_types = ["METAL", "NONE"]
    elif platform in {"linux", "linux2", "win32"}:
        preferred_device_types = ["OPTIX", "CUDA", "NONE"]
    else:
        preferred_device_types = ["NONE"]

    for device_type in preferred_device_types:
        try:
            _set_cycles_device(device_type)
            return device_type
        except TypeError:
            continue
    _set_cycles_device("NONE")
    return "NONE"


def _enable_auto_smooth_for_weighted_normals():
    for obj in bpy.data.objects:
        if obj.type != "MESH" or obj.data is None:
            continue
        if any(mod.type == "WEIGHTED_NORMAL" for mod in obj.modifiers):
            try:
                obj.data.use_auto_smooth = True
            except AttributeError:
                # Blender 4.x removed Mesh.use_auto_smooth; best-effort fallback.
                pass


def _render_with_fallback():
    try:
        bpy.ops.render.render(write_still=True)
    except RuntimeError as exc:
        message = str(exc).lower()
        if bpy.context.scene.cycles.device != "GPU":
            raise
        if "failed to create cuda context" not in message and "out of memory" not in message:
            raise
        print("Cycles GPU initialization failed; retrying render on CPU.")
        _set_cycles_device("NONE")
        bpy.ops.render.render(write_still=True)

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

    bpy.context.scene.render.engine = 'CYCLES'
    _configure_cycles_device()

    # Setting up rendering resolution
    bpy.context.scene.render.resolution_x = 512
    bpy.context.scene.render.resolution_y = 512

    # Set max samples to 1024
    bpy.context.scene.cycles.samples = 512

    # Set color mode to RGB
    bpy.context.scene.render.image_settings.color_mode = 'RGB'

    # Some generated edits trigger depsgraph updates before rendering.
    _enable_auto_smooth_for_weighted_normals()

    # Read and execute the code from the specified file
    with open(code_fpath, "r") as f:
        code = f.read()
    try:
        exec(code)
    except Exception as exc:
        raise RuntimeError(f"Generated edit execution failed: {exc}") from exc

    _enable_auto_smooth_for_weighted_normals()

    # Render from camera1
    if 'Camera' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render.png')
        _render_with_fallback()

    # Render from camera1
    if 'Camera1' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera1']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render1.png')
        _render_with_fallback()

    # Render from camera2
    if 'Camera2' in bpy.data.objects:
        bpy.context.scene.camera = bpy.data.objects['Camera2']
        bpy.context.scene.render.image_settings.file_format = 'PNG'
        bpy.context.scene.render.filepath = os.path.join(rendering_dir, 'render2.png')
        _render_with_fallback()
