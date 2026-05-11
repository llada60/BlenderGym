"""
Reconstruct an intermediate_metadata_oneshot_*.json from an existing outputs directory.

Usage:
    python reconstruct_metadata.py --outputs_dir system/outputs/outputs_material_05-10-23-55-43
    python reconstruct_metadata.py --outputs_dir system/outputs/outputs_material_05-10-23-55-43 \
        --generator_type claude-code-sonnet-4-6 \
        --bench_data_dir bench_data \
        --info_saving_dir info_saved
"""

import argparse
import json
import os
import re


TASK_INSTANCE_COUNT = {
    'geometry': 55,
    'material': 45,
    'blendshape': 85,
    'placement': 50,
    'lighting': 45,
}

# Maps task name (in dir/bench_data) to internal system task type
TASK_TRANSLATE = {
    'geometry': 'geonodes',
    'material': 'material',
    'blendshape': 'shapekey',
    'placement': 'placement',
    'lighting': 'lighting',
}

VARIANT = 'tune_leap'


def detect_task_from_dir_name(dir_name):
    """Infer task from outputs directory basename, e.g. outputs_material_05-10-23-55-43."""
    for task in TASK_INSTANCE_COUNT:
        if f'_{task}_' in dir_name or dir_name.endswith(f'_{task}'):
            return task
    return None


def detect_task_from_instances(outputs_dir):
    """Infer task by looking at the first subdirectory name."""
    for entry in sorted(os.listdir(outputs_dir)):
        for task in TASK_INSTANCE_COUNT:
            if entry.startswith(task):
                return task
    return None


def load_iteration_json(iter_path):
    with open(iter_path) as f:
        return json.load(f)


def build_instance_record(outputs_dir, task_instance_id, bench_data_dir):
    """
    Returns (record_dict, error_str).
    error_str is None on success; record_dict is always populated with at least the bench paths.
    """
    bench_instance_dir = os.path.abspath(os.path.join(bench_data_dir, task_instance_id))
    base_record = {
        'instance_dir_path': bench_instance_dir,
        'blender_file_path': os.path.join(bench_instance_dir, 'blender_file.blend'),
        'start_script_path': os.path.join(bench_instance_dir, 'start.py'),
        'goal_script_path': os.path.join(bench_instance_dir, 'goal.py'),
    }

    instance_output_dir = os.path.join(outputs_dir, task_instance_id)
    if not os.path.isdir(instance_output_dir):
        return {**base_record, 'error': 'output directory missing'}, 'output directory missing'

    output_base = os.path.join(instance_output_dir, 'instance0', f'{VARIANT}_d1_b1')
    scripts_dir = os.path.join(output_base, 'scripts')
    renders_dir = os.path.join(output_base, 'renders')
    iter_path = os.path.join(output_base, 'thought_process', 'iteration_0.json')

    missing = []
    for p, label in [(scripts_dir, 'scripts'), (renders_dir, 'renders'), (iter_path, 'iteration_0.json')]:
        if not os.path.exists(p):
            missing.append(label)
    if missing:
        err = f'missing: {", ".join(missing)}'
        return {**base_record, 'error': err}, err

    proposal_edits_paths = sorted(
        os.path.join(scripts_dir, f) for f in os.listdir(scripts_dir) if f.endswith('.py')
    )
    proposal_renders_paths = sorted(
        os.path.join(renders_dir, f) for f in os.listdir(renders_dir) if f.endswith('.png')
    )

    if not proposal_edits_paths or not proposal_renders_paths:
        err = 'scripts or renders directory is empty'
        return {**base_record, 'error': err}, err

    try:
        info = load_iteration_json(iter_path)
        winner = info[-1]
        selected_edit_path = 'system/' + winner['winner_code']
        selected_render_path = 'system/' + winner['winner_image']
    except Exception as exc:
        err = f'could not parse iteration_0.json: {exc}'
        return {**base_record, 'error': err}, err

    record = {
        **base_record,
        'proposal_edits_paths': proposal_edits_paths,
        'proposal_renders_paths': proposal_renders_paths,
        'selected_edit_path': selected_edit_path,
        'selected_render_path': selected_render_path,
    }
    return record, None


def main():
    parser = argparse.ArgumentParser(
        description='Reconstruct intermediate_metadata_oneshot JSON from an existing outputs directory.'
    )
    parser.add_argument(
        '--outputs_dir',
        required=True,
        help='Path to the outputs directory, e.g. system/outputs/outputs_material_05-10-23-55-43',
    )
    parser.add_argument(
        '--generator_type',
        default='unknown',
        help='Model/generator identifier to embed in the metadata (default: unknown)',
    )
    parser.add_argument(
        '--bench_data_dir',
        default='bench_data',
        help='Path to bench_data directory (default: bench_data)',
    )
    parser.add_argument(
        '--info_saving_dir',
        default='info_saved',
        help='Directory where output JSON files are written (default: info_saved)',
    )
    args = parser.parse_args()

    outputs_dir = args.outputs_dir.rstrip('/')
    dir_name = os.path.basename(outputs_dir)

    # Detect task
    task = detect_task_from_dir_name(dir_name) or detect_task_from_instances(outputs_dir)
    if task is None:
        raise ValueError(
            f'Cannot detect task from directory name "{dir_name}" or its contents. '
            'Expected one of: ' + ', '.join(TASK_INSTANCE_COUNT.keys())
        )

    # Detect timestamp suffix from dir name, e.g. "05-10-23-55-43"
    ts_match = re.search(r'(\d{2}-\d{2}-\d{2}-\d{2}-\d{2})$', dir_name)
    timestamp = ts_match.group(1) if ts_match else 'reconstructed'

    # Determine the full instance list from bench_data
    total = TASK_INSTANCE_COUNT[task]
    all_instance_ids = [f'{task}{i}' for i in range(1, total + 1)]

    # Detect which instances actually exist in the outputs dir
    present_in_output = set(os.listdir(outputs_dir))

    results = {}
    failed_instances = []

    for instance_id in all_instance_ids:
        if instance_id not in present_in_output:
            bench_instance_dir = os.path.abspath(os.path.join(args.bench_data_dir, instance_id))
            record = {
                'instance_dir_path': bench_instance_dir,
                'blender_file_path': os.path.join(bench_instance_dir, 'blender_file.blend'),
                'start_script_path': os.path.join(bench_instance_dir, 'start.py'),
                'goal_script_path': os.path.join(bench_instance_dir, 'goal.py'),
                'error': 'no output directory found',
            }
            results[instance_id] = record
            failed_instances.append({
                'task': task,
                'task_instance_id': instance_id,
                'instance_dir_path': bench_instance_dir,
                'error': 'no output directory found',
            })
            continue

        record, err = build_instance_record(outputs_dir, instance_id, args.bench_data_dir)
        results[instance_id] = record
        if err is not None:
            failed_instances.append({
                'task': task,
                'task_instance_id': instance_id,
                'instance_dir_path': record['instance_dir_path'],
                'error': err,
            })

    status = 'completed_with_failures' if failed_instances else 'completed'

    generation_results = {
        'output_dir_name': dir_name,
        'generator_type': args.generator_type,
        'tree_dims': [1, 1],
        'mode': 'oneshot_no_verifier',
        'status': status,
        task: results,
    }
    if failed_instances:
        generation_results['failed_instances'] = failed_instances

    os.makedirs(args.info_saving_dir, exist_ok=True)

    metadata_path = os.path.join(
        args.info_saving_dir,
        f'intermediate_metadata_oneshot_{task}_{timestamp}.json',
    )
    errors_path = os.path.join(
        args.info_saving_dir,
        f'errors_oneshot_{task}_{timestamp}.json',
    )

    with open(metadata_path, 'w') as f:
        json.dump(generation_results, f, indent=4)
    print(f'Wrote metadata -> {metadata_path}')

    if failed_instances:
        with open(errors_path, 'w') as f:
            json.dump(failed_instances, f, indent=4)
        print(f'Wrote {len(failed_instances)} error(s) -> {errors_path}')
    else:
        print('No errors.')

    print(f'Done: {len(results) - len(failed_instances)}/{len(all_instance_ids)} instances succeeded.')


if __name__ == '__main__':
    main()
