"""
One-shot entrypoint: generate exactly one runnable edit without verifier selection.
"""

from pathlib import Path
import argparse
import sys
import yaml

from refinement_process import refinement_oneshot_no_verifier, is_response_limit_error
from tasksolver.keychain import KeyChain


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="BlenderAlchemy one-shot arguments")
    parser.add_argument("--starter_blend", type=str, required=True, help="Path to the base blender file.")
    parser.add_argument("--blender_base", type=str, required=True, help="Blender render script path.")
    parser.add_argument("--blender_script", type=str, required=True, help="Starter bpy script path.")
    parser.add_argument("--config", type=str, required=True, help="Path to yaml config file.")

    args = parser.parse_args()

    with open(args.config) as stream:
        config = yaml.safe_load(stream)

    kc = KeyChain()
    for el in config["credentials"]:
        if config["credentials"][el] is not None:
            kc.add_key(el, config["credentials"][el])

    output_dir = Path(config["output"]["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    variants = [el.strip() for el in config["run_config"]["variants"]]
    if len(variants) != 1:
        raise ValueError("main_oneshot.py expects exactly one variant.")

    try:
        result = refinement_oneshot_no_verifier(
            config,
            credentials=kc,
            blender_file=args.starter_blend,
            blender_script=args.blender_base,
            init_code=args.blender_script,
            method_variation=variants[0],
            output_folder=output_dir / "instance0" / f"{variants[0]}_d1_b1",
        )
    except Exception as exc:
        if is_response_limit_error(exc):
            print(f"FATAL_LLM_RESPONSE_LIMIT: {exc}", file=sys.stderr)
            sys.exit(42)
        raise

    print(result)
