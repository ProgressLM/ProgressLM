
import argparse
import json
import os
import subprocess
import tempfile
import shutil
import collections
from pathlib import Path

def build_dataset_index(jsonl_path, image_root_dir):
    """
    Reads the .jsonl file and builds a structured index in memory.
    The index makes it efficient to find trajectories and their components.
    """
    print("Building dataset index...")
    index = collections.defaultdict(dict)
    with open(jsonl_path, 'r') as f:
        for line in f:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
            except json.JSONDecodeError:
                print(f"Warning: Skipping malformed JSON line: {line.strip()}")
                continue

            parts = data['id'].split('/')
            if len(parts) < 3:
                print(f"Warning: Skipping line with unexpected ID format: {data['id']}")
                continue
            
            task_type = "/".join(parts[:-1])
            timestamp_id = parts[-1]

            # Resolve image paths
            def resolve_paths(img_names):
                # Handle cases where img_names might be a string instead of a list
                if isinstance(img_names, str):
                    img_names = [img_names]
                # The ID in the dataset already contains the sub-path
                base_path = os.path.join(image_root_dir, data['id'])
                return [os.path.join(base_path, name) for name in img_names]

            if timestamp_id not in index[task_type]:
                index[task_type][timestamp_id] = {
                    "task_goal": data["task_goal"],
                    "total_steps": int(data["total_steps"]),
                    "visual_demo_paths": resolve_paths(data["visual_demo"]),
                    "stages": []
                }
            
            # Add stage info
            progress_score_str = data["progress_score"].replace('%', '')
            index[task_type][timestamp_id]["stages"].append({
                "image_path": resolve_paths(data["stage_to_estimate"])[0],
                "progress": float(progress_score_str)
            })
    
    print(f"Index built. Found {len(index)} task types.")
    return index

def find_self_reference(trajectory_data):
    """Returns the trajectory's own visual demo."""
    return trajectory_data["visual_demo_paths"]

def find_cross_reference(current_task_type, current_timestamp_id, dataset_index):
    """
    Finds a reference trajectory from a different instance based on user-defined rules.
    """
    search_pool = dataset_index[current_task_type]
    current_trajectory = search_pool[current_timestamp_id]

    # Priority 1: Find another trajectory with the exact same task_goal
    for timestamp, trajectory_data in search_pool.items():
        if timestamp == current_timestamp_id:
            continue
        if trajectory_data["task_goal"] == current_trajectory["task_goal"]:
            print(f"  Found cross-ref for {current_timestamp_id} (same task_goal): {timestamp}")
            return trajectory_data["visual_demo_paths"]

    # Priority 2: Find another trajectory with the same number of total_steps
    for timestamp, trajectory_data in search_pool.items():
        if timestamp == current_timestamp_id:
            continue
        if trajectory_data["total_steps"] == current_trajectory["total_steps"]:
            print(f"  Found cross-ref for {current_timestamp_id} (same total_steps): {timestamp}")
            return trajectory_data["visual_demo_paths"]

    # Fallback: If no suitable cross-reference is found, use its own demo
    print(f"  Warning: No suitable cross-ref found for {current_timestamp_id}. Falling back to self-reference.")
    return current_trajectory["visual_demo_paths"]

def prepare_temp_dir(image_paths, temp_dir_root):
    """Creates a temporary directory and copies image files into it."""
    temp_dir = tempfile.mkdtemp(dir=temp_dir_root)
    for img_path in image_paths:
        if not os.path.exists(img_path):
            print(f"Warning: Image file not found, skipping: {img_path}")
            continue
        shutil.copy(img_path, temp_dir)
    return temp_dir

def main():
    parser = argparse.ArgumentParser(description="Main pipeline to evaluate VLAC based on a .jsonl dataset.")
    parser.add_argument('--jsonl_path', type=str, required=True, help="Path to the .jsonl dataset file.")
    parser.add_argument('--image_root_dir', type=str, required=True, help="Root directory where all trajectory image folders are stored.")
    parser.add_argument('--model_path', type=str, required=True, help="Path to the VLAC model directory.")
    parser.add_argument('--output_dir', type=str, default='./results_pipeline', help="Directory to save evaluation results.")
    parser.add_argument('--cross_trajectory_ref', action='store_true', help="Enable cross-trajectory reference finding.")
    parser.add_argument('--gpu_ids', type=str, default="0", help="Comma-separated list of GPU IDs to use (e.g., '0,1,4').")

    # Capture any other arguments to pass them through to run_eval.py
    args, passthrough_args = parser.parse_known_args()

    # Ensure output directory exists
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Create a root for all temporary directories for this run
    run_temp_root = tempfile.mkdtemp(prefix="vlac_eval_")
    print(f"Using temporary root for image data: {run_temp_root}")

    try:
        dataset_index = build_dataset_index(args.jsonl_path, args.image_root_dir)
        
        script_dir = Path(__file__).parent
        vlac_eval_script = script_dir / 'run_eval.py'

        total_trajectories = sum(len(timestamps) for timestamps in dataset_index.values())
        print(f"Starting evaluation for {total_trajectories} trajectories...")
        
        count = 0
        for task_type, trajectories in dataset_index.items():
            for timestamp_id, trajectory_data in trajectories.items():
                count += 1
                print(f"\n--- ({count}/{total_trajectories}) Processing Trajectory: {task_type}/{timestamp_id} ---")

                # 1. Prepare Main Trajectory (sorted by progress)
                stages = sorted(trajectory_data["stages"], key=lambda x: x["progress"])
                main_trajectory_paths = [s["image_path"] for s in stages]
                
                if not main_trajectory_paths:
                    print("Warning: No stages found for this trajectory. Skipping.")
                    continue

                # 2. Select Reference Trajectory
                if args.cross_trajectory_ref:
                    ref_trajectory_paths = find_cross_reference(task_type, timestamp_id, dataset_index)
                else:
                    ref_trajectory_paths = find_self_reference(trajectory_data)

                if not ref_trajectory_paths:
                    print("Warning: No reference trajectory found. Skipping.")
                    continue

                # 3. Prepare temporary directories and data
                main_dir = prepare_temp_dir(main_trajectory_paths, run_temp_root)
                ref_dir = prepare_temp_dir(ref_trajectory_paths, run_temp_root)
                
                # 4. Execute run_eval.py
                output_filename = f"{task_type.replace('/', '_')}_{timestamp_id}.json"
                
                command = [
                    'python', str(vlac_eval_script),
                    '--model_path', args.model_path,
                    '--data_dir', main_dir,
                    '--ref_dir', ref_dir,
                    '--task', trajectory_data['task_goal'],
                    '--output_dir', args.output_dir,
                    '--output_name', output_filename,
                    '--gpu_ids', args.gpu_ids
                ] + passthrough_args

                print(f"Executing VLAC evaluation for {timestamp_id}...")
                # print(f"  Command: {' '.join(command)}") # For debugging

                try:
                    process = subprocess.run(command, check=True, capture_output=True, text=True)
                    print(f"  Successfully evaluated. Results saved in {args.output_dir}/{output_filename}")
                    # print(process.stdout) # For detailed output
                except subprocess.CalledProcessError as e:
                    print(f"  ERROR: Evaluation failed for {timestamp_id}.")
                    print(f"  Return Code: {e.returncode}")
                    print(f"  Stdout: {e.stdout}")
                    print(f"  Stderr: {e.stderr}")

    finally:
        # 5. Cleanup
        print(f"\nCleaning up temporary directory: {run_temp_root}")
        shutil.rmtree(run_temp_root)

if __name__ == "__main__":
    main()
