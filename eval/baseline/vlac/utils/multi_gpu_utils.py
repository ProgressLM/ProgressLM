"""
Multi-GPU Data Parallel Inference Utilities for VLAC

This module provides multi-GPU data parallel inference capabilities for VLAC model evaluation.
It uses multiprocessing to distribute workload across multiple GPUs, with each GPU running
an independent model instance.

Usage:
    from multi_gpu_utils import multi_gpu_trajectory_critic

    critic_list, value_list = multi_gpu_trajectory_critic(
        model_path="path/to/model",
        task="Pick up the bowl",
        image_list=images,
        num_gpus=4,
        batch_num=5
    )
"""

import os
import sys
import torch
import multiprocessing as mp
from typing import List, Optional, Tuple, Any
from PIL import Image
import numpy as np


def _worker_process(
    gpu_id: int,
    model_path: str,
    model_type: str,
    task: str,
    image_indices: List[int],
    shared_image_dir: str,
    ref_image_indices: Optional[List[int]],
    batch_num: int,
    ref_num: int,
    skip: int,
    temperature: float,
    top_k: int,
    rich: bool,
    think: bool,
    reverse_eval: bool,
    frame_skip: bool,
    result_queue: mp.Queue,
):
    """
    Worker process function that runs on a single GPU.

    Args:
        gpu_id: GPU device ID (0, 1, 2, ...)
        model_path: Path to VLAC model
        model_type: Model type (e.g., 'internvl2')
        task: Task description
        image_indices: List of image indices this worker should process
        shared_image_dir: Directory containing serialized images
        ref_image_indices: Reference image indices (if any)
        batch_num: Batch size for inference
        ref_num: Number of reference images to use
        skip: Frame skip step
        temperature: Sampling temperature
        top_k: Top-k sampling
        rich: Enable rich mode
        think: Enable Chain-of-Thought
        reverse_eval: Enable reverse evaluation
        frame_skip: Enable frame skip mode
        result_queue: Queue to put results
    """
    try:
        # Import here to avoid issues with multiprocessing
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from model_utils import GAC_model

        # Set device for this worker
        device = f"cuda:{gpu_id}"
        print(f"[GPU {gpu_id}] Starting worker on {device}")
        print(f"[GPU {gpu_id}] Processing {len(image_indices)} image pairs")

        # Load images from shared directory
        image_list = []
        for idx in sorted(set(image_indices)):
            img_path = os.path.join(shared_image_dir, f"image_{idx}.pkl")
            import pickle
            with open(img_path, 'rb') as f:
                img = pickle.load(f)
            image_list.append(img)

        # Load reference images if provided
        ref_image_list = None
        if ref_image_indices is not None:
            ref_image_list = []
            for idx in ref_image_indices:
                img_path = os.path.join(shared_image_dir, f"ref_image_{idx}.pkl")
                import pickle
                with open(img_path, 'rb') as f:
                    img = pickle.load(f)
                ref_image_list.append(img)

        # Initialize model on this GPU
        critic = GAC_model(tag='critic')
        critic.init_model(
            model_path=model_path,
            model_type=model_type,
            device_map=device
        )
        critic.temperature = temperature
        critic.top_k = top_k
        critic.set_config()
        critic.set_system_prompt()

        print(f"[GPU {gpu_id}] Model initialized successfully")

        # Run trajectory critic
        critic_list, value_list = critic.get_trajectory_critic(
            task=task,
            image_list=image_list,
            ref_image_list=ref_image_list,
            batch_num=batch_num,
            ref_num=ref_num,
            think=think,
            skip=skip,
            rich=rich,
            reverse_eval=reverse_eval,
            frame_skip=frame_skip
        )

        print(f"[GPU {gpu_id}] Processing completed. Generated {len(critic_list)} critic scores")

        # Put results in queue with original indices
        result_queue.put({
            'gpu_id': gpu_id,
            'image_indices': image_indices,
            'critic_list': critic_list,
            'value_list': value_list,
            'success': True
        })

    except Exception as e:
        import traceback
        error_msg = f"[GPU {gpu_id}] Error: {str(e)}\n{traceback.format_exc()}"
        print(error_msg)
        result_queue.put({
            'gpu_id': gpu_id,
            'image_indices': image_indices,
            'success': False,
            'error': error_msg
        })


def _save_images_to_temp(image_list: List[Image.Image], temp_dir: str, prefix: str = "image"):
    """
    Save images to temporary directory for sharing across processes.

    Args:
        image_list: List of PIL images
        temp_dir: Temporary directory path
        prefix: Prefix for saved files

    Returns:
        List of indices
    """
    import pickle
    indices = []
    for i, img in enumerate(image_list):
        img_path = os.path.join(temp_dir, f"{prefix}_{i}.pkl")
        with open(img_path, 'wb') as f:
            pickle.dump(img, f)
        indices.append(i)
    return indices


def multi_gpu_trajectory_critic(
    model_path: str,
    model_type: str,
    task: str,
    image_list: List[Image.Image],
    num_gpus: int,
    ref_image_list: Optional[List[Image.Image]] = None,
    batch_num: int = 5,
    ref_num: int = 6,
    skip: int = 1,
    temperature: float = 0.5,
    top_k: int = 1,
    rich: bool = False,
    think: bool = False,
    reverse_eval: bool = False,
    frame_skip: bool = True,
) -> Tuple[List[str], List[float]]:
    """
    Multi-GPU data parallel trajectory critic evaluation.

    This function splits the image sequence across multiple GPUs and processes
    them in parallel. Each GPU runs an independent model instance.

    Args:
        model_path: Path to VLAC model directory
        model_type: Model type (e.g., 'internvl2')
        task: Task description
        image_list: List of PIL images to evaluate
        num_gpus: Number of GPUs to use
        ref_image_list: Optional reference trajectory images
        batch_num: Batch size for inference on each GPU
        ref_num: Number of reference images to sample
        skip: Frame skip step for pair-wise evaluation
        temperature: Sampling temperature
        top_k: Top-k sampling
        rich: Enable rich mode (decimal values)
        think: Enable Chain-of-Thought reasoning
        reverse_eval: Enable reverse evaluation
        frame_skip: Enable frame skip mode

    Returns:
        critic_list: List of critic scores
        value_list: List of progress values (0-100)
    """

    # Validate num_gpus
    available_gpus = torch.cuda.device_count()
    if num_gpus > available_gpus:
        print(f"Warning: Requested {num_gpus} GPUs but only {available_gpus} available. Using {available_gpus} GPUs.")
        num_gpus = available_gpus

    if num_gpus < 1:
        raise ValueError(f"num_gpus must be >= 1, got {num_gpus}")

    print(f"\n{'='*80}")
    print(f"Multi-GPU Data Parallel Inference")
    print(f"{'='*80}")
    print(f"Number of GPUs: {num_gpus}")
    print(f"Total images: {len(image_list)}")
    print(f"Skip: {skip}")

    # Calculate number of evaluation steps
    if frame_skip:
        select_idx = list(range(skip, len(image_list), skip))
    else:
        select_idx = list(range(skip, len(image_list)))

    total_steps = len(select_idx)
    print(f"Total evaluation steps: {total_steps}")
    print(f"Steps per GPU: ~{total_steps // num_gpus}")
    print(f"{'='*80}\n")

    # Create temporary directory for sharing images
    import tempfile
    import shutil
    temp_dir = tempfile.mkdtemp(prefix="vlac_multi_gpu_")

    try:
        # Save images to temp directory
        print("Preparing shared image data...")
        image_indices = _save_images_to_temp(image_list, temp_dir, "image")

        ref_image_indices = None
        if ref_image_list is not None:
            ref_image_indices = _save_images_to_temp(ref_image_list, temp_dir, "ref_image")

        # Split work across GPUs
        # We need to ensure each GPU gets a contiguous chunk of the original image_list
        # because critic evaluation needs sequential context

        # Strategy: Split the original image list into num_gpus chunks
        # Each chunk will process its own segment independently
        images_per_gpu = len(image_list) // num_gpus
        gpu_image_ranges = []

        for gpu_id in range(num_gpus):
            start_idx = gpu_id * images_per_gpu
            if gpu_id == num_gpus - 1:
                # Last GPU takes remaining images
                end_idx = len(image_list)
            else:
                end_idx = (gpu_id + 1) * images_per_gpu
                # Add overlap for context (need previous frame for comparison)
                end_idx = min(end_idx + skip, len(image_list))

            gpu_image_ranges.append((start_idx, end_idx))

        # Create result queue
        ctx = mp.get_context('spawn')
        result_queue = ctx.Queue()

        # Start worker processes
        processes = []
        for gpu_id in range(num_gpus):
            start_idx, end_idx = gpu_image_ranges[gpu_id]
            worker_image_indices = list(range(start_idx, end_idx))

            p = ctx.Process(
                target=_worker_process,
                args=(
                    gpu_id,
                    model_path,
                    model_type,
                    task,
                    worker_image_indices,
                    temp_dir,
                    ref_image_indices,
                    batch_num,
                    ref_num,
                    skip,
                    temperature,
                    top_k,
                    rich,
                    think,
                    reverse_eval,
                    frame_skip,
                    result_queue,
                )
            )
            p.start()
            processes.append(p)
            print(f"Started worker process for GPU {gpu_id} (images {start_idx}-{end_idx})")

        # Collect results
        print("\nWaiting for workers to complete...")
        results = []
        for _ in range(num_gpus):
            result = result_queue.get()
            results.append(result)
            if result['success']:
                print(f"[GPU {result['gpu_id']}] Completed successfully")
            else:
                print(f"[GPU {result['gpu_id']}] Failed with error:")
                print(result['error'])

        # Wait for all processes to finish
        for p in processes:
            p.join()

        # Check for errors
        errors = [r for r in results if not r['success']]
        if errors:
            raise RuntimeError(f"Multi-GPU inference failed on {len(errors)} GPU(s)")

        # Merge results in order
        print("\nMerging results from all GPUs...")

        # Sort results by GPU ID to maintain order
        results.sort(key=lambda x: x['gpu_id'])

        # Concatenate critic and value lists
        merged_critic_list = []
        merged_value_list = []

        for result in results:
            merged_critic_list.extend(result['critic_list'])
            merged_value_list.extend(result['value_list'])

        print(f"Merged {len(merged_critic_list)} critic scores from {num_gpus} GPUs")
        print(f"Multi-GPU inference completed successfully!\n")

        return merged_critic_list, merged_value_list

    finally:
        # Clean up temp directory
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)
            print(f"Cleaned up temporary directory: {temp_dir}")


if __name__ == "__main__":
    # Test example
    print("Multi-GPU utilities module loaded successfully")
    print(f"Available GPUs: {torch.cuda.device_count()}")
