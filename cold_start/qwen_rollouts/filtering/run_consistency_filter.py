import os
import sys
import json
import argparse
import time
from tqdm import tqdm
from typing import List, Dict, Any
import torch
import traceback
import multiprocessing as mp
from multiprocessing import Manager, Process, Queue

# Local imports
from worldvlm_dataset import load_worldvlm_dataset
from worldvlm_prompt import build_worldvlm_prompt_from_item, WORLDVLM_CONSISTENCY_SYSTEM_PROMPT
from qwen2_vl.model import Qwen2VLChat


def parse_consistency_response(response: str) -> bool:
    """
    Parse the model's yes/no response.

    Args:
        response: Model output string

    Returns:
        True if 'yes', False if 'no' or uncertain
    """
    response = response.strip().lower()

    # Direct match
    if response == 'yes':
        return True
    elif response == 'no':
        return False

    # Fuzzy match for robustness
    if 'yes' in response and 'no' not in response:
        return True
    elif 'no' in response and 'yes' not in response:
        return False

    # Default to False for unclear responses
    return False


def worker_process(gpu_id: int, data_slice: List, args, progress_queue: Queue, result_queue: Queue):
    """Worker process for one GPU with batch inference."""

    # Set this process to use only one GPU
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    # Create GPU-specific output file
    gpu_output_file = args.output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')

    try:
        # Initialize model on this GPU
        model = Qwen2VLChat(
            model_path=args.model_path,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            max_new_tokens=args.max_new_tokens,
            use_custom_prompt=False,
            system_prompt=WORLDVLM_CONSISTENCY_SYSTEM_PROMPT,
            min_pixels=args.min_pixels,
            max_pixels=args.max_pixels,
            verbose=False
        )

        # Process data in batches
        batch_size = args.batch_size
        results = []
        processed_count = 0

        i = 0
        while i < len(data_slice):
            batch_end = min(i + batch_size, len(data_slice))
            batch_items = data_slice[i:batch_end]

            try:
                # Build batch prompts
                batch_messages = []
                for item in batch_items:
                    messages = build_worldvlm_prompt_from_item(
                        item,
                        min_pixels=args.min_pixels,
                        max_pixels=args.max_pixels
                    )
                    batch_messages.append(messages)

                # Batch inference
                batch_responses = model.generate(batch_messages)

                # Process responses
                for item, response in zip(batch_items, batch_responses):
                    source_id = item.get('source_id', '')

                    try:
                        is_consistent = parse_consistency_response(response)

                        result = {
                            "source_id": source_id,
                            "data_source": item.get('data_source', ''),
                            "start_img": item['start_img'],
                            "end_img": item['end_img'],
                            "action": item['action'],
                            "task_goal": item.get('task_goal', ''),
                            "step": item.get('step', ''),
                            "model_response": response,
                            "is_consistent": is_consistent,
                            "timestamp": time.time(),
                            "gpu_id": gpu_id
                        }
                        results.append(result)

                        # Report progress: (processed_count, passed, failed, errors)
                        progress_queue.put((1, 1 if is_consistent else 0, 0 if is_consistent else 1, 0))

                    except Exception as e:
                        # Parse error
                        result = {
                            "source_id": source_id,
                            "data_source": item.get('data_source', ''),
                            "start_img": item.get('start_img', ''),
                            "end_img": item.get('end_img', ''),
                            "action": item.get('action', ''),
                            "task_goal": item.get('task_goal', ''),
                            "step": item.get('step', ''),
                            "model_response": response if response else "",
                            "is_consistent": False,
                            "error": str(e),
                            "timestamp": time.time(),
                            "gpu_id": gpu_id
                        }
                        results.append(result)
                        progress_queue.put((1, 0, 0, 1))

            except Exception as e:
                # Batch error
                for item in batch_items:
                    source_id = item.get('source_id', '')
                    result = {
                        "source_id": source_id,
                        "data_source": item.get('data_source', ''),
                        "start_img": item.get('start_img', ''),
                        "end_img": item.get('end_img', ''),
                        "action": item.get('action', ''),
                        "task_goal": item.get('task_goal', ''),
                        "step": item.get('step', ''),
                        "model_response": "",
                        "is_consistent": False,
                        "error": f"Batch error: {str(e)}",
                        "timestamp": time.time(),
                        "gpu_id": gpu_id
                    }
                    results.append(result)
                    progress_queue.put((1, 0, 0, 1))

            # Update processed count
            processed_count += len(batch_items)

            # Save results immediately after each batch (so main process can merge anytime)
            with open(gpu_output_file, 'w') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

            i = batch_end

        # Final save
        with open(gpu_output_file, 'w') as f:
            for res in results:
                f.write(json.dumps(res, ensure_ascii=False) + '\n')

        # Explicitly clean up model and CUDA resources before sending results
        try:
            del model
            torch.cuda.empty_cache()
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception as e:
            print(f"GPU {gpu_id}: Error during cleanup: {e}")

        # Send results back
        result_queue.put((gpu_id, results))

    except Exception as e:
        print(f"GPU {gpu_id} worker failed: {e}")
        traceback.print_exc()
        result_queue.put((gpu_id, []))


def run_consistency_filtering(args):
    """Run consistency filtering with multi-GPU batch inference."""

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")
    image_root = args.image_root if hasattr(args, 'image_root') and args.image_root else None

    import sys
    from io import StringIO

    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    data = load_worldvlm_dataset(args.dataset_path, image_root=image_root)

    if not args.verbose:
        sys.stdout = old_stdout

    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} samples")

    # Get GPU configuration
    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    num_gpus = len(gpu_ids)

    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Total samples: {len(data)}")
    print(f"Batch size per GPU: {args.batch_size}")
    print(f"Total batches: {(len(data) + args.batch_size - 1) // args.batch_size}")

    # Split data across GPUs
    samples_per_gpu = len(data) // num_gpus
    data_slices = []
    for i in range(num_gpus):
        start_idx = i * samples_per_gpu
        if i == num_gpus - 1:
            end_idx = len(data)
        else:
            end_idx = start_idx + samples_per_gpu
        data_slices.append(data[start_idx:end_idx])
        print(f"GPU {gpu_ids[i]}: processing samples {start_idx}-{end_idx-1} ({len(data_slices[i])} samples)")

    # Create output directory
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    output_file = args.output_file

    # Create queues for progress and results
    progress_queue = Queue()
    result_queue = Queue()

    # Start worker processes
    print(f"\nStarting {num_gpus} worker processes...")
    processes = []
    for i, gpu_id in enumerate(gpu_ids):
        p = Process(target=worker_process, args=(gpu_id, data_slices[i], args, progress_queue, result_queue))
        p.start()
        processes.append(p)
        print(f"  Started GPU {gpu_id} worker (PID: {p.pid})")

    print(f"\nProcessing {len(data)} samples with {num_gpus} GPUs (batch_size={args.batch_size} per GPU)...\n")

    # Monitor progress with unified tqdm
    total_processed = 0
    passed_count = 0
    failed_count = 0
    error_count = 0

    # Use tqdm with fixed width and disable dynamic resizing
    pbar = tqdm(total=len(data), desc="Filtering", ncols=100,
                miniters=1, mininterval=1.0, smoothing=0.1,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]')

    # Monitor progress from all workers
    last_stat_print = 0
    last_update_time = time.time()
    update_interval = 1.0  # Update tqdm every 1 second (smoother)
    accumulated_updates = 0
    interrupted = False
    last_progress_time = time.time()
    no_progress_timeout = 60  # 60 seconds without progress triggers timeout check

    try:
        while total_processed < len(data):
            # Check if all workers are done
            all_workers_done = all(not p.is_alive() for p in processes)
            if all_workers_done:
                # All workers finished, break out of loop
                break

            # Collect all pending progress updates
            batch_proc_count = 0
            batch_passed = 0
            batch_failed = 0
            batch_errors = 0

            while not progress_queue.empty():
                # Each progress update: (processed_count, passed, failed, errors)
                proc_count, passed, failed, errors = progress_queue.get_nowait()
                batch_proc_count += proc_count
                batch_passed += passed
                batch_failed += failed
                batch_errors += errors

            # Update counters
            if batch_proc_count > 0:
                total_processed += batch_proc_count
                passed_count += batch_passed
                failed_count += batch_failed
                error_count += batch_errors
                accumulated_updates += batch_proc_count
                last_progress_time = time.time()  # Reset timeout timer

            # Check for timeout (no progress for too long)
            if time.time() - last_progress_time > no_progress_timeout:
                all_workers_done = all(not p.is_alive() for p in processes)
                if all_workers_done:
                    pbar.write(f"\nNo progress for {no_progress_timeout}s and all workers finished. Exiting loop.")
                    break

            # Update tqdm periodically (not every sample)
            current_time = time.time()
            if accumulated_updates > 0 and (current_time - last_update_time >= update_interval or total_processed >= len(data)):
                pbar.update(accumulated_updates)
                pass_rate = passed_count / total_processed * 100 if total_processed > 0 else 0
                pbar.set_postfix_str(f"Yes={passed_count}, Rate={pass_rate:.1f}%, Err={error_count}")
                accumulated_updates = 0
                last_update_time = current_time

                # Save merged results every SAVE_INTERVAL samples (silently, don't print)
                if total_processed - last_stat_print >= args.save_interval:
                    last_stat_print = total_processed

                    # Merge and save intermediate results from all GPUs
                    merged_results = []
                    for gpu_id in gpu_ids:
                        gpu_file = output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')
                        if os.path.exists(gpu_file):
                            with open(gpu_file, 'r') as f:
                                for line in f:
                                    if line.strip():
                                        merged_results.append(json.loads(line))

                    if merged_results:
                        merged_results.sort(key=lambda x: x.get('source_id', ''))
                        with open(output_file, 'w') as f:
                            for res in merged_results:
                                f.write(json.dumps(res, ensure_ascii=False) + '\n')

            # Small sleep to avoid busy waiting
            time.sleep(0.2)

    except KeyboardInterrupt:
        interrupted = True
        pbar.close()

        print("\n\n" + "=" * 70)
        print("⚠️  用户中断 (Ctrl+C) - 正在停止所有GPU进程...")
        print("=" * 70)

        # Terminate all processes
        print("发送终止信号到所有worker进程...")
        for i, p in enumerate(processes):
            if p.is_alive():
                print(f"  停止 GPU {gpu_ids[i]} worker (PID: {p.pid})")
                p.terminate()

        # Wait for graceful shutdown (max 5 seconds)
        print("等待进程优雅退出 (最多5秒)...")
        start_wait = time.time()
        all_terminated = False
        while time.time() - start_wait < 5:
            if all(not p.is_alive() for p in processes):
                all_terminated = True
                break
            time.sleep(0.1)

        # Force kill if still running
        if not all_terminated:
            print("强制终止未响应的进程...")
            for i, p in enumerate(processes):
                if p.is_alive():
                    print(f"  强制终止 GPU {gpu_ids[i]} worker (PID: {p.pid})")
                    p.kill()
                    p.join(timeout=1)

        print("\n所有worker进程已停止")

        # Collect partial results
        print("收集已完成的部分结果...")
        all_results = []
        while not result_queue.empty():
            try:
                gpu_id, results = result_queue.get(timeout=0.5)
                print(f"  从 GPU {gpu_id} 收集到 {len(results)} 个结果")
                all_results.extend(results)
            except:
                break

        if all_results:
            all_results.sort(key=lambda x: x.get('source_id', ''))
            partial_file = output_file.replace('.jsonl', '_partial.jsonl')
            print(f"\n保存部分结果到: {partial_file}")
            with open(partial_file, 'w') as f:
                for res in all_results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

            print(f"\n已保存 {len(all_results)} 个部分结果")
            print("=" * 70)

        # Exit cleanly
        sys.exit(130)  # Standard exit code for Ctrl+C

    pbar.close()

    # Wait for all workers to complete
    if not interrupted:
        # First, drain any remaining progress updates before waiting for workers
        print("\nDraining remaining progress updates...")
        drain_count = 0
        drain_passed = 0
        drain_failed = 0
        drain_errors = 0
        while not progress_queue.empty():
            try:
                proc_count, passed, failed, errors = progress_queue.get_nowait()
                total_processed += proc_count
                passed_count += passed
                failed_count += failed
                error_count += errors
                drain_count += proc_count
                drain_passed += passed
                drain_failed += failed
                drain_errors += errors
            except:
                break

        if drain_count > 0:
            print(f"Drained {drain_count} remaining progress updates")
            pass_rate = passed_count / total_processed * 100 if total_processed > 0 else 0
            print(f"Final count: {total_processed}/{len(data)}, Passed={passed_count}, Rate={pass_rate:.1f}%")

        print("\nWaiting for all workers to finish...")

        # Wait for workers while draining result_queue to prevent blocking
        all_done = False
        wait_start = time.time()
        max_wait = 300  # Max 5 minutes wait

        while not all_done and (time.time() - wait_start < max_wait):
            all_done = all(not p.is_alive() for p in processes)

            # Drain result_queue while waiting to prevent queue full blocking
            while not result_queue.empty():
                try:
                    result_queue.get_nowait()
                except:
                    break

            if not all_done:
                time.sleep(0.5)

        if not all_done:
            print(f"\nWarning: Some workers didn't finish after {max_wait}s, forcing termination...")
            for i, p in enumerate(processes):
                if p.is_alive():
                    print(f"  Terminating GPU {gpu_ids[i]} worker (PID: {p.pid})")
                    p.terminate()
                    p.join(timeout=5)
                    if p.is_alive():
                        print(f"  Force killing GPU {gpu_ids[i]} worker")
                        p.kill()

        # Final join to clean up
        for p in processes:
            if p.is_alive():
                p.join(timeout=1)

    # Collect results from all workers
    pbar.write("Collecting results from all GPUs...")
    all_results = []
    while not result_queue.empty():
        gpu_id, results = result_queue.get()
        pbar.write(f"  Received {len(results)} results from GPU {gpu_id}")
        all_results.extend(results)

    # Sort results by source_id
    all_results.sort(key=lambda x: x.get('source_id', ''))

    # Write final results
    pbar.write(f"Writing {len(all_results)} results to {output_file}...")
    with open(output_file, 'w') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    # Print final summary
    print("\n" + "=" * 70)
    print("CONSISTENCY FILTERING SUMMARY")
    print("=" * 70)
    print(f"Total samples: {len(data)}")
    print(f"Passed (consistent): {passed_count} ({passed_count/len(data)*100:.2f}%)")
    print(f"Failed (inconsistent): {failed_count} ({failed_count/len(data)*100:.2f}%)")
    print(f"Errors: {error_count} ({error_count/len(data)*100:.2f}%)")
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    # Save summary
    summary_file = output_file.replace('.jsonl', '_summary.json')
    summary = {
        "total_samples": len(data),
        "passed": passed_count,
        "failed": failed_count,
        "errors": error_count,
        "pass_rate": passed_count / len(data) if len(data) > 0 else 0,
        "batch_size": args.batch_size,
        "num_gpus": num_gpus,
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "output_file": args.output_file
    }
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(description="WorldVLM Consistency Filtering")

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the Qwen2-VL model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the WorldVLM dataset (JSONL format)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path for results")

    # Optional arguments
    parser.add_argument("--image-root", type=str, default=None,
                        help="Root directory to prepend to relative image paths")
    parser.add_argument("--batch-size", type=int, default=32,
                        help="Batch size for inference (default: 32)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of samples to process (-1 for all)")
    parser.add_argument("--save-interval", type=int, default=100,
                        help="Save intermediate results every N samples")

    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.01,
                        help="Sampling temperature")
    parser.add_argument("--top-p", type=float, default=0.001,
                        help="Top-p sampling parameter")
    parser.add_argument("--top-k", type=int, default=1,
                        help="Top-k sampling parameter")
    parser.add_argument("--max-new-tokens", type=int, default=32,
                        help="Maximum number of tokens to generate")

    # Image processing parameters
    parser.add_argument("--min-pixels", type=int, default=1280*28*28,
                        help="Minimum pixels for image processing")
    parser.add_argument("--max-pixels", type=int, default=5120*28*28,
                        help="Maximum pixels for image processing")

    # Misc
    parser.add_argument("--verbose", action="store_true",
                        help="Print detailed output")

    args = parser.parse_args()

    # Validate paths
    if not os.path.exists(args.dataset_path):
        print(f"Error: Dataset file not found: {args.dataset_path}")
        sys.exit(1)

    if not os.path.exists(args.model_path):
        print(f"Error: Model path not found: {args.model_path}")
        sys.exit(1)

    # Run filtering
    run_consistency_filtering(args)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
