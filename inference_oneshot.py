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
    # 'lighting': 50
}


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
        default=f"{os.path.abspath('infinigen/Blender.app/Contents/MacOS/Blender')}",
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

    args = parser.parse_args()

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
    info_saving_json_path = os.path.join(
        args.info_saving_dir_path,
        f'intermediate_metadata_oneshot_{time.strftime("%m-%d-%H-%M-%S")}.json'
    )

    if 'all' in tasks:
        task_instance_dir_paths = {task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task] + 1)] for task in task_instance_count_dict.keys()}
    elif 'test' in tasks:
        task_instance_dir_paths = {task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, 4)] for task in task_instance_count_dict.keys()}
    elif 'subset' in tasks:
        task_instance_dir_paths = {task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, 11)] for task in task_instance_count_dict.keys()}
    else:
        task_instance_dir_paths = {task: [os.path.join('bench_data', f'{task}{i}') for i in range(1, task_instance_count_dict[task] + 1)] for task in tasks}

    print(f'task_instance_dir_paths: {task_instance_dir_paths}')

    starter_time = time.strftime("%m-%d-%H-%M-%S")
    generation_results = {
        "output_dir_name": f"outputs_{starter_time}",
        'generator_type': args.generator_type,
        'tree_dims': (1, 1),
        'mode': 'oneshot_no_verifier',
    }

    for task, instance_dir_paths in task_instance_dir_paths.items():
        generation_results[task] = {}

        for instance_dir_path in instance_dir_paths:
            task_instance_id = os.path.basename(instance_dir_path)
            generation_results[task][task_instance_id] = {}

            instance_dir_path = os.path.abspath(instance_dir_path)
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
                )
            except Exception as exc:
                generation_results[task][task_instance_id]['error'] = str(exc)
                with open(info_saving_json_path, 'w') as file:
                    json.dump(generation_results, file, indent=4)
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

            with open(info_saving_json_path, 'w') as file:
                json.dump(generation_results, file, indent=4)
