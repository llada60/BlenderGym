import os
import argparse
import time
import json

from utils import BlenderAlchemy_run_oneshot


task_instance_count_dict = {
    'geometry': 55,
    'material': 45,
    'blendshape': 85,
    'placement': 50,
    'lighting': 45,
}


def build_task_instance_dir_paths(tasks):
    if 'all' in tasks:
        return {
            task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task] + 1)]
            for task in task_instance_count_dict.keys()
        }
    if 'test' in tasks:
        return {
            task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, 4)]
            for task in task_instance_count_dict.keys()
        }
    if 'subset' in tasks:
        return {
            task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, 11)]
            for task in task_instance_count_dict.keys()
        }
    return {
        task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task] + 1)]
        for task in tasks
    }


def build_pending_items(task_instance_dir_paths):
    pending = []
    for task, instance_dir_paths in task_instance_dir_paths.items():
        for instance_dir_path in instance_dir_paths:
            pending.append({
                "task": task,
                "instance_dir_path": os.path.abspath(instance_dir_path),
            })
    return pending


def save_json(path, payload):
    with open(path, 'w') as file:
        json.dump(payload, file, indent=4)


def persist_resume_state(resume_state_path, task_signature, starter_time, output_dir_name, info_saving_json_path, pending_items, generation_results):
    resume_state = {
        "task_signature": task_signature,
        "starter_time": starter_time,
        "output_dir_name": output_dir_name,
        "info_saving_json_path": info_saving_json_path,
        "pending_items": pending_items,
        "generation_results": generation_results,
    }
    save_json(resume_state_path, resume_state)


def normalize_task_tag(task_arg):
    return task_arg.strip().replace(',', '_').replace(' ', '')


def normalize_tree_dims(tree_dims):
    if isinstance(tree_dims, str):
        parts = tree_dims.lower().split('x')
        if len(parts) == 2 and all(part.isdigit() for part in parts):
            return [int(parts[0]), int(parts[1])]
        return tree_dims
    if isinstance(tree_dims, (list, tuple)) and len(tree_dims) == 2:
        return [int(tree_dims[0]), int(tree_dims[1])]
    return tree_dims


def should_stop_on_error(exc):
    msg = str(exc)
    return "LIMIT" in msg


def resolve_resume_state_path(info_saving_dir_path, task_name, generator_type):
    canonical_path = os.path.join(
        info_saving_dir_path,
        f"oneshot_resume_{task_name}_{generator_type}.json"
    )
    legacy_path = os.path.join(
        info_saving_dir_path,
        f"oneshot_resume_{generator_type}_{task_name}.json"
    )

    # Prefer the legacy generator-first filename when it exists, because
    # existing historical runs in this workspace use that layout.
    if os.path.exists(legacy_path):
        return legacy_path
    return canonical_path


def run_single_task(args, task):
    task_tag = normalize_task_tag(task)
    task_instance_dir_paths = build_task_instance_dir_paths([task])
    pending_items = build_pending_items(task_instance_dir_paths)

    print(f'task_instance_dir_paths: {task_instance_dir_paths}')

    task_signature = {
        "task": task,
        "task_tag": task_tag,
        "generator_type": args.generator_type,
        "tree_dims": normalize_tree_dims(args.tree_dims),
        "blender_render_script_path": os.path.abspath(args.blender_render_script_path),
        "infinigen_installation_path": os.path.abspath(args.infinigen_installation_path),
    }
    resume_state_path = resolve_resume_state_path(
        args.info_saving_dir_path,
        task,
        args.generator_type,
    )

    if os.path.exists(resume_state_path):
        with open(resume_state_path, 'r') as file:
            resume_state = json.load(file)

        resume_signature = resume_state.get("task_signature", {})
        normalized_resume_signature = {
            "task": resume_signature.get("task"),
            "task_tag": resume_signature.get("task_tag", normalize_task_tag(resume_signature.get("task", ""))),
            "generator_type": resume_signature.get("generator_type"),
            "tree_dims": normalize_tree_dims(resume_signature.get("tree_dims")),
            "blender_render_script_path": os.path.abspath(resume_signature.get("blender_render_script_path", "")),
            "infinigen_installation_path": os.path.abspath(resume_signature.get("infinigen_installation_path", "")),
        }
        if normalized_resume_signature != task_signature:
            raise ValueError(
                f"Found resume state at {resume_state_path}, but it does not match the current arguments. "
                "Use the same command to resume, or remove that resume file first."
            )

        starter_time = resume_state["starter_time"]
        output_dir_name = resume_state.get("output_dir_name", f"outputs_{starter_time}")
        info_saving_json_path = resume_state["info_saving_json_path"]
        pending_items = resume_state["pending_items"]
        generation_results = resume_state["generation_results"]
        print(f"Resuming interrupted one-shot inference from {resume_state_path}")
    else:
        starter_time = time.strftime("%m-%d-%H-%M-%S")
        output_dir_name = f"outputs_{task_tag}_{starter_time}"
        info_saving_json_path = os.path.join(
            args.info_saving_dir_path,
            f'intermediate_metadata_oneshot_{task_tag}_{starter_time}.json'
        )
        generation_results = {
            "output_dir_name": output_dir_name,
            "generator_type": args.generator_type,
            "tree_dims": [1, 1],
            "mode": "oneshot_no_verifier",
            "status": "running",
            "resume_state_path": resume_state_path,
        }

    save_json(info_saving_json_path, generation_results)
    persist_resume_state(
        resume_state_path,
        task_signature,
        starter_time,
        output_dir_name,
        info_saving_json_path,
        pending_items,
        generation_results,
    )

    for pending_index, pending_item in enumerate(pending_items):
        instance_dir_path = pending_item["instance_dir_path"]
        task_instance_id = os.path.basename(instance_dir_path)

        generation_results.setdefault(task, {})
        generation_results[task].setdefault(task_instance_id, {})

        blender_file_path = os.path.join(instance_dir_path, 'blender_file.blend')
        start_file_path = os.path.join(instance_dir_path, 'start.py')
        start_render_path = os.path.join(instance_dir_path, 'renders/start')
        goal_file_path = os.path.join(instance_dir_path, 'goal.py')
        goal_render_path = os.path.join(instance_dir_path, 'renders/goal')

        try:
            proposal_edits_paths, proposal_renders_paths, selected_edit_path, selected_render_path = BlenderAlchemy_run_oneshot(
                blender_file_path,
                start_file_path,
                start_render_path,
                goal_render_path,
                args.blender_render_script_path,
                task_instance_id,
                task,
                args.infinigen_installation_path,
                args.generator_type,
                starter_time=starter_time,
                output_dir_name=output_dir_name,
            )
        except Exception as exc:
            generation_results[task][task_instance_id]['instance_dir_path'] = instance_dir_path
            generation_results[task][task_instance_id]['blender_file_path'] = blender_file_path
            generation_results[task][task_instance_id]['start_script_path'] = start_file_path
            generation_results[task][task_instance_id]['goal_script_path'] = goal_file_path
            generation_results[task][task_instance_id]['error'] = str(exc)
            if should_stop_on_error(exc):
                generation_results["status"] = "paused_on_error"
                generation_results["last_error"] = {
                    "task": task,
                    "task_instance_id": task_instance_id,
                    "instance_dir_path": instance_dir_path,
                    "error": str(exc),
                }

                remaining_pending_items = pending_items[pending_index:]
                save_json(info_saving_json_path, generation_results)
                persist_resume_state(
                    resume_state_path,
                    task_signature,
                    starter_time,
                    output_dir_name,
                    info_saving_json_path,
                    remaining_pending_items,
                    generation_results,
                )
                raise RuntimeError(
                    f"One-shot inference stopped on {task_instance_id}: {exc}\n"
                    f"Resume state saved to {resume_state_path}"
                ) from exc

            generation_results.setdefault("failed_instances", [])
            generation_results["failed_instances"].append({
                "task": task,
                "task_instance_id": task_instance_id,
                "instance_dir_path": instance_dir_path,
                "error": str(exc),
            })
            generation_results["status"] = "running_with_failures"
            generation_results["last_error"] = generation_results["failed_instances"][-1]

            remaining_pending_items = pending_items[pending_index + 1:]
            save_json(info_saving_json_path, generation_results)
            persist_resume_state(
                resume_state_path,
                task_signature,
                starter_time,
                output_dir_name,
                info_saving_json_path,
                remaining_pending_items,
                generation_results,
            )
            print(
                f"Skipping failed instance {task_instance_id}: {exc}\n"
                f"Resume state updated at {resume_state_path}"
            )
            continue

        print(f'proposal_edits_paths: {proposal_edits_paths}')
        print(f'proposal_renders_paths: {proposal_renders_paths}')
        print(f'selected_edit_path: {selected_edit_path}')
        print(f'selected_render_path: {selected_render_path}')

        generation_results[task][task_instance_id]['instance_dir_path'] = instance_dir_path
        generation_results[task][task_instance_id]['blender_file_path'] = blender_file_path
        generation_results[task][task_instance_id]['start_script_path'] = start_file_path
        generation_results[task][task_instance_id]['goal_script_path'] = goal_file_path
        generation_results[task][task_instance_id]['proposal_edits_paths'] = proposal_edits_paths
        generation_results[task][task_instance_id]['proposal_renders_paths'] = proposal_renders_paths
        generation_results[task][task_instance_id]['selected_edit_path'] = selected_edit_path
        generation_results[task][task_instance_id]['selected_render_path'] = selected_render_path
        save_json(info_saving_json_path, generation_results)
        persist_resume_state(
            resume_state_path,
            task_signature,
            starter_time,
            output_dir_name,
            info_saving_json_path,
            pending_items[pending_index + 1:],
            generation_results,
        )

    if generation_results.get("failed_instances"):
        generation_results["status"] = "completed_with_failures"
    else:
        generation_results["status"] = "completed"
    save_json(info_saving_json_path, generation_results)
    persist_resume_state(
        resume_state_path,
        task_signature,
        starter_time,
        output_dir_name,
        info_saving_json_path,
        [],
        generation_results,
    )


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='One-shot image-based program edits without verifier')

    parser.add_argument(
        '--task',
        type=str,
        default='test',
        help='`all`, `test`, `subset`, or comma-separated list of: `material`, `geometry`, `blendshape`, `placement`, `lighting`'
    )

    parser.add_argument(
        '--info_saving_dir_path',
        type=str,
        default='info_saved',
        help='Directory where intermediate inference metadata is saved.'
    )

    parser.add_argument(
        '--blender_render_script_path',
        type=str,
        default=f"{os.path.abspath(os.path.join('bench_data', 'pipeline_render_script.py'))}",
        help='The Blender render script.'
    )

    parser.add_argument(
        '--infinigen_installation_path',
        type=str,
        default=f"{os.path.abspath('infinigen/blender/blender')}",
        help='The installation path of blender executable file.'
    )

    parser.add_argument(
        '--generator_type',
        type=str,
        required=True,
        help='model_id of the one-shot VLM generator.'
    )

    parser.add_argument(
        '--tree_dims',
        type=str,
        default='1x1',
        help='Must be `1x1` for one-shot inference; kept only for explicitness.'
    )

    parser.add_argument(
        '--render_device',
        type=str,
        choices=['auto', 'cpu', 'gpu'],
        default='auto',
        help="Cycles render device selection. Use 'cpu' to skip GPU initialization entirely.",
    )

    args = parser.parse_args()

    if args.render_device == 'cpu':
        os.environ["BLENDERGYM_FORCE_CPU"] = "1"
    elif args.render_device == 'gpu':
        os.environ.pop("BLENDERGYM_FORCE_CPU", None)

    if args.tree_dims != '1x1':
        raise ValueError('inference_oneshot.py only supports --tree_dims 1x1.')

    tasks = args.task.strip().split(',')
    if len(tasks) == 0:
        raise ValueError('Invalid input for --task: no task input detected.')

    for task in tasks:
        if task not in ('all', 'test', 'subset') and task not in task_instance_count_dict.keys():
            raise ValueError(f'Invalid input for --task: {task} is not a valid input for "--task".')

    if not os.path.isfile(args.blender_render_script_path):
        raise ValueError(f'Invalid input for blender_render_script_path: {args.blender_render_script_path}')

    if not os.path.exists(args.infinigen_installation_path):
        raise ValueError(f'Invalid input for infinigen_installation_path: {args.infinigen_installation_path}')

    os.makedirs(args.info_saving_dir_path, exist_ok=True)

    concrete_tasks = [task for task in tasks if task in task_instance_count_dict]
    special_modes = [task for task in tasks if task in ('all', 'test', 'subset')]

    if len(special_modes) > 0 and len(tasks) > 1:
        raise ValueError('Do not mix `all`, `test`, or `subset` with concrete task names.')

    if len(special_modes) == 1:
        expanded_tasks = list(build_task_instance_dir_paths(tasks).keys())
        for task in expanded_tasks:
            run_single_task(args, task)
    else:
        for task in concrete_tasks:
            run_single_task(args, task)
