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
from multiprocessing import Process, Queue

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Local imports
from clean_text_dataset import load_clean_text_dataset
from clean_text_prompt import build_clean_text_prompt_from_item, CLEAN_TEXT_SYSTEM_PROMPT
from text_format_validator import is_sample_format_valid
from qwen2_vl.model import Qwen2VLChat


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
            system_prompt=CLEAN_TEXT_SYSTEM_PROMPT,
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
                    messages = build_clean_text_prompt_from_item(item)
                    batch_messages.append(messages)

                # Batch inference
                batch_responses = model.generate(batch_messages)

                # Process responses
                for item, response in zip(batch_items, batch_responses):
                    try:
                        # Format validation
                        validation_sample = {
                            'id': item['id'],
                            'text_demo': response,
                            'total_steps': item['total_steps']
                        }
                        is_valid, errors = is_sample_format_valid(validation_sample)

                        result = {
                            "id": item['id'],
                            "new_text_demo": response,
                            "error": False,
                            "format_error": not is_valid
                        }

                        # Optionally include detailed error messages if verbose
                        if not is_valid and args.verbose:
                            result["format_errors"] = errors

                        results.append(result)

                        # Report progress: (processed_count, error_count, format_error_count)
                        progress_queue.put((1, 0, 1 if not is_valid else 0))

                    except Exception as e:
                        # Parse error for this specific item
                        result = {
                            "id": item['id'],
                            "new_text_demo": response if response else "",
                            "error": True,
                            "format_error": True,
                            "error_message": f"Processing error: {str(e)}"
                        }
                        results.append(result)
                        progress_queue.put((1, 1, 1))

            except Exception as e:
                # Batch error - mark all items in batch as errors
                for item in batch_items:
                    result = {
                        "id": item['id'],
                        "new_text_demo": "",
                        "error": True,
                        "format_error": True,
                        "error_message": f"Batch error: {str(e)}"
                    }
                    results.append(result)
                    progress_queue.put((1, 1, 1))

            # Update processed count
            processed_count += len(batch_items)

            # Save results immediately after each batch
            with open(gpu_output_file, 'w', encoding='utf-8') as f:
                for res in results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

            i = batch_end

        # Final save
        with open(gpu_output_file, 'w', encoding='utf-8') as f:
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


def run_clean_text_inference(args):
    """Run text cleaning inference with multi-GPU batch inference."""

    # Load dataset
    print(f"Loading dataset from {args.dataset_path}")

    import sys
    from io import StringIO

    if not args.verbose:
        old_stdout = sys.stdout
        sys.stdout = StringIO()

    data = load_clean_text_dataset(
        args.dataset_path,
        num_inferences=args.num_inferences
    )

    if not args.verbose:
        sys.stdout = old_stdout

    if args.limit > 0:
        data = data[:args.limit]
        print(f"Limited to first {args.limit} samples (after expansion)")

    # Get GPU configuration
    gpu_ids = [int(x) for x in os.environ.get('CUDA_VISIBLE_DEVICES', '0').split(',')]
    num_gpus = len(gpu_ids)

    print(f"Using {num_gpus} GPUs: {gpu_ids}")
    print(f"Total samples (expanded): {len(data)}")
    if args.num_inferences > 1:
        print(f"Original samples: {len(data) // args.num_inferences}")
        print(f"Inferences per sample: {args.num_inferences}")
    print(f"Batch size per GPU: {args.batch_size}")

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
    processing_errors = 0
    format_errors = 0

    # Use tqdm with fixed width - dynamic single-line update only
    pbar = tqdm(total=len(data), desc="Processing", ncols=140,
                miniters=1, mininterval=0.5, smoothing=0.3, dynamic_ncols=False,
                bar_format='{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}')

    # Monitor progress from all workers
    last_update_time = time.time()
    update_interval = 0.5  # Update every 0.5 seconds for more responsive display
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
            batch_processing_errors = 0
            batch_format_errors = 0

            while not progress_queue.empty():
                # Each progress update: (processed_count, error_count, format_error_count)
                proc_count, error_count, format_error_count = progress_queue.get_nowait()
                batch_proc_count += proc_count
                batch_processing_errors += error_count
                batch_format_errors += format_error_count

            # Update counters
            if batch_proc_count > 0:
                total_processed += batch_proc_count
                processing_errors += batch_processing_errors
                format_errors += batch_format_errors
                accumulated_updates += batch_proc_count
                last_progress_time = time.time()  # Reset timeout timer

            # Check for timeout (no progress for too long)
            if time.time() - last_progress_time > no_progress_timeout:
                all_workers_done = all(not p.is_alive() for p in processes)
                if all_workers_done:
                    pbar.write(f"\nNo progress for {no_progress_timeout}s and all workers finished. Exiting loop.")
                    break

            # Update tqdm periodically
            current_time = time.time()
            if accumulated_updates > 0 and (current_time - last_update_time >= update_interval or total_processed >= len(data)):
                pbar.update(accumulated_updates)

                # Calculate rates
                error_rate = processing_errors / total_processed * 100 if total_processed > 0 else 0.0
                format_error_rate = format_errors / total_processed * 100 if total_processed > 0 else 0.0
                format_accuracy = 100.0 - format_error_rate

                # Update postfix with all metrics in one line
                pbar.set_postfix_str(
                    f"Error: {error_rate:.1f}% | FormatAcc: {format_accuracy:.1f}%",
                    refresh=True
                )

                accumulated_updates = 0
                last_update_time = current_time

            # Small sleep to avoid busy waiting
            time.sleep(0.1)

    except KeyboardInterrupt:
        interrupted = True
        pbar.close()

        print("\n\n" + "=" * 70)
        print("WARNING: User interrupt (Ctrl+C) - Stopping all GPU processes...")
        print("=" * 70)

        # Terminate all processes
        print("Sending termination signal to all worker processes...")
        for i, p in enumerate(processes):
            if p.is_alive():
                print(f"  Stopping GPU {gpu_ids[i]} worker (PID: {p.pid})")
                p.terminate()

        # Wait for graceful shutdown (max 5 seconds)
        print("Waiting for processes to exit gracefully (max 5 seconds)...")
        start_wait = time.time()
        all_terminated = False
        while time.time() - start_wait < 5:
            if all(not p.is_alive() for p in processes):
                all_terminated = True
                break
            time.sleep(0.1)

        # Force kill if still running
        if not all_terminated:
            print("Force killing unresponsive processes...")
            for i, p in enumerate(processes):
                if p.is_alive():
                    print(f"  Force killing GPU {gpu_ids[i]} worker (PID: {p.pid})")
                    p.kill()
                    p.join(timeout=1)

        print("\nAll worker processes stopped")

        # Collect partial results
        print("Collecting partial results...")
        all_results = []
        while not result_queue.empty():
            try:
                gpu_id, results = result_queue.get(timeout=0.5)
                print(f"  Collected {len(results)} results from GPU {gpu_id}")
                all_results.extend(results)
            except:
                break

        if all_results:
            all_results.sort(key=lambda x: x.get('id', ''))
            partial_file = output_file.replace('.jsonl', '_partial.jsonl')
            print(f"\nSaving partial results to: {partial_file}")
            with open(partial_file, 'w', encoding='utf-8') as f:
                for res in all_results:
                    f.write(json.dumps(res, ensure_ascii=False) + '\n')

            print(f"\nSaved {len(all_results)} partial results")
            print("=" * 70)

        # Exit cleanly
        sys.exit(130)

    pbar.close()

    # Wait for all workers to complete
    if not interrupted:
        # First, drain any remaining progress updates before waiting for workers
        print("\nDraining remaining progress updates...")
        drain_count = 0
        while not progress_queue.empty():
            try:
                proc_count, error_count, format_error_count = progress_queue.get_nowait()
                total_processed += proc_count
                processing_errors += error_count
                format_errors += format_error_count
                drain_count += proc_count
            except:
                break

        if drain_count > 0:
            print(f"Drained {drain_count} remaining progress updates")
            error_rate = processing_errors / total_processed * 100 if total_processed > 0 else 0.0
            format_accuracy = 100.0 - (format_errors / total_processed * 100 if total_processed > 0 else 0.0)
            print(f"Final count: {total_processed}/{len(data)}, Error: {error_rate:.1f}%, FormatAcc: {format_accuracy:.1f}%")

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

    # Sleep 10 seconds before merging results
    print("\nAll GPU workers finished. Sleeping for 10 seconds before merging results...")
    time.sleep(10)

    # Merge results from all GPU files
    print("Merging results from all GPU files...")
    all_results = []
    for gpu_id in gpu_ids:
        gpu_file = output_file.replace('.jsonl', f'_gpu{gpu_id}.jsonl')
        if os.path.exists(gpu_file):
            with open(gpu_file, 'r', encoding='utf-8') as f:
                gpu_results = []
                for line in f:
                    if line.strip():
                        gpu_results.append(json.loads(line))
                print(f"  Loaded {len(gpu_results)} results from GPU {gpu_id}")
                all_results.extend(gpu_results)
        else:
            print(f"  Warning: GPU {gpu_id} file not found: {gpu_file}")

    # Sort results by id
    all_results.sort(key=lambda x: x.get('id', ''))

    # Write merged results to final output file
    print(f"\nWriting {len(all_results)} merged results to {output_file}...")
    with open(output_file, 'w', encoding='utf-8') as f:
        for res in all_results:
            f.write(json.dumps(res, ensure_ascii=False) + '\n')

    print(f"Merged results saved. Individual GPU files are preserved.")

    # Calculate final statistics
    error_samples = sum(1 for r in all_results if r['error'])
    format_error_samples = sum(1 for r in all_results if r['format_error'])
    valid_samples = len(all_results) - format_error_samples

    # Print final summary
    print("\n" + "=" * 70)
    print("TEXT CLEANING SUMMARY")
    print("=" * 70)
    print(f"Total samples (expanded): {len(data)}")
    if args.num_inferences > 1:
        print(f"Original samples: {len(data) // args.num_inferences}")
        print(f"Inferences per sample: {args.num_inferences}")
    print(f"Processed: {len(all_results)}")
    print(f"Processing errors: {error_samples} ({error_samples/len(all_results)*100:.2f}%)")
    print(f"Format errors: {format_error_samples} ({format_error_samples/len(all_results)*100:.2f}%)")
    print(f"Valid samples: {valid_samples} ({valid_samples/len(all_results)*100:.2f}%)")
    print(f"Results saved to: {output_file}")
    print("=" * 70)

    # Save summary
    summary_file = output_file.replace('.jsonl', '_summary.json')
    summary = {
        "total_samples_expanded": len(data),
        "original_samples": len(data) // args.num_inferences if args.num_inferences > 1 else len(data),
        "num_inferences_per_sample": args.num_inferences,
        "processed": len(all_results),
        "processing_errors": error_samples,
        "format_errors": format_error_samples,
        "valid_samples": valid_samples,
        "error_rate": error_samples / len(all_results) if all_results else 0,
        "format_error_rate": format_error_samples / len(all_results) if all_results else 0,
        "valid_rate": valid_samples / len(all_results) if all_results else 0,
        "batch_size": args.batch_size,
        "num_gpus": num_gpus,
        "dataset_path": args.dataset_path,
        "model_path": args.model_path,
        "output_file": args.output_file
    }
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print(f"Summary saved to: {summary_file}\n")


def main():
    parser = argparse.ArgumentParser(description="Text Cleaning - Batch Inference")

    # Required arguments
    parser.add_argument("--model-path", type=str, required=True,
                        help="Path to the Qwen2-VL model")
    parser.add_argument("--dataset-path", type=str, required=True,
                        help="Path to the dataset (JSONL format)")
    parser.add_argument("--output-file", type=str, required=True,
                        help="Output JSONL file path for results")

    # Optional arguments
    parser.add_argument("--batch-size", type=int, default=16,
                        help="Batch size for inference per GPU (default: 16)")
    parser.add_argument("--num-inferences", type=int, default=1,
                        help="Number of inferences per sample (data expansion factor, default: 1)")
    parser.add_argument("--limit", type=int, default=-1,
                        help="Limit number of samples to process after expansion (-1 for all)")

    # Model parameters
    parser.add_argument("--temperature", type=float, default=0.3,
                        help="Sampling temperature (default: 0.3 for consistent cleaning)")
    parser.add_argument("--top-p", type=float, default=0.9,
                        help="Top-p sampling parameter (default: 0.9)")
    parser.add_argument("--top-k", type=int, default=50,
                        help="Top-k sampling parameter (default: 50)")
    parser.add_argument("--max-new-tokens", type=int, default=2048,
                        help="Maximum number of tokens to generate (default: 2048)")

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

    # Run inference
    run_clean_text_inference(args)


if __name__ == "__main__":
    # Set multiprocessing start method
    mp.set_start_method('spawn', force=True)
    main()
