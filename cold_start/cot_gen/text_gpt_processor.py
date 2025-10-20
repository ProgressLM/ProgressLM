#!/usr/bin/env python3
"""
GPT-5-mini Real-time API Processor for Text-Based Task Progress Evaluation

This version works with text demonstrations instead of visual demonstrations.
It processes samples with text_demo format and evaluates progress based on 
textual step descriptions combined with a single current state image.

Data format:
{
    "id": "sample_id",
    "task_goal": "inserting a battery...",
    "text_demo": ["step1", "step2", "step3"],
    "total_steps": 3,
    "stage_to_estimate": "image.jpg",
    "closest_idx": 1,
    "progress_score": "33%",
    "data_source": "source_name"
}
"""

import json
import os
import sys
import time
import base64
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime
import argparse
from tqdm import tqdm
import traceback
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Global lock for file writing
write_lock = threading.Lock()

class TextDemoProgressProcessor:
    """Text-based progress evaluation processor"""
    
    def __init__(self, api_key: str, image_dir: str, model: str = "gpt-5-mini"):
        """
        Initialize processor
        
        Args:
            api_key: OpenAI API key
            image_dir: Base directory for images
            model: Model to use
        """
        self.client = OpenAI(api_key=api_key)
        self.model = model
        self.image_dir = Path(image_dir)
        
        if not self.image_dir.exists():
            raise ValueError(f"Image directory does not exist: {image_dir}")
    
    def encode_image(self, image_path: Path) -> str:
        """
        Encode image to base64
        
        Args:
            image_path: Path to image
        
        Returns:
            Base64 encoded image string
        """
        try:
            with open(image_path, "rb") as image_file:
                return base64.b64encode(image_file.read()).decode('utf-8')
        except Exception as e:
            raise ValueError(f"Cannot read image {image_path}: {str(e)}")
    
    def build_image_content(self, image_path: Path) -> Dict:
        """
        Build image message content
        
        Args:
            image_path: Image path
        
        Returns:
            OpenAI API formatted image content
        """
        base64_image = self.encode_image(image_path)
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}",
                "detail": "high"
            }
        }
    
    def calculate_step_progress(self, step_number: int, total_steps: int) -> str:
        """
        Calculate progress percentage for a given step
        
        Args:
            step_number: Current step number (1-indexed)
            total_steps: Total number of steps
        
        Returns:
            Progress percentage as string
        """
        if total_steps == 0:
            return "0%"
        progress = int((step_number / total_steps) * 100)
        return f"{progress}%"
    
    def build_message_content(self, sample: Dict) -> List[Dict]:
        """
        Build complete message content for text-based demonstrations
        
        Args:
            sample: A sample from JSONL
        
        Returns:
            Message content list
        """
        content = []
        
        # 1. System prompt
        system_prompt = (
            "You are an expert AI analyst specializing in visual task-progress evaluations. "
            "Your objective is not to estimate from scratch. "
            "Instead, your task is to construct a perfect, human-like chain of thought that "
            "logically explains and justifies a known, ground-truth progress score. "
            "Your entire response must read as if you are deducing the conclusion independently "
            "from visual analysis alone.\n\n"
            "This is the system prompt for normal inference. You are a progress estimator that "
            "evaluates the progress of an ongoing task based on a textual demonstration of its "
            "step-by-step progression. The demonstration consists of a sequence of text instructions "
            "(text_demo), each describing one step of the process. Each step explicitly states the "
            "corresponding progress value (ranging from 0% to 100%), showing how the task evolves "
            "from start to completion.\n\n"
            "Here is the demonstration:"
        )
        content.append({"type": "text", "text": system_prompt})
        
        # 2. Add task goal
        task_goal_text = f"\n\nOur goal is {sample['task_goal']}.\n\n"
        content.append({"type": "text", "text": task_goal_text})
        
        # 3. Add text demonstrations with progress
        text_demos = sample['text_demo']
        total_steps = len(text_demos)
        
        text_demo_content = ""
        for i, demo_text in enumerate(text_demos, 1):
            progress = self.calculate_step_progress(i, total_steps)
            text_demo_content += f"Step {i}: {demo_text}. The Progress for now is {progress}.\n"
        
        content.append({"type": "text", "text": text_demo_content})
        
        # 4. Add current state prompt
        content.append({
            "type": "text",
            "text": "\nHere is the current state that you need to estimate:"
        })
        
        # 5. Add stage_to_estimate image
        sample_id = sample['id']
        stage_image = sample['stage_to_estimate']
        
        # Handle both single string and list format for stage_to_estimate
        if isinstance(stage_image, list):
            stage_image = stage_image[0]
        
        stage_path = self.image_dir / sample_id / stage_image
        if not stage_path.exists():
            raise FileNotFoundError(f"Evaluation image does not exist: {stage_path}")
        content.append(self.build_image_content(stage_path))
        
        # 6. Add critical rule and ground truth
        critical_rule = (
            "\n\n**Critical Rule** The correct final progress score will be provided to you. "
            "However, you must **never** reveal or imply that you already know the answer. "
            "Your reasoning must appear as a fully original, independent visual analysis "
            "derived from the images.\n\n"
            f"**Ground-Truth Progress Result**\n"
            f"Closest Reference Frame: The No. {sample['closest_idx']} text demo is the most relevant one\n"
            f"Final Progress Score to Justify: {sample['progress_score']}"
        )
        content.append({"type": "text", "text": critical_rule})
        
        # 7. Add task instructions and output format
        task_instructions = (
            "\n\nYour task:\n"
            "1. Analyze the text_demo to understand how the task visually and conceptually "
            "progresses from start to completion.\n"
            "2. Identify the step from the text_demo that are most visually and semantically "
            "similar to the current state image.\n"
            "3. Compare the current state image with the chosen reference step to determine "
            "whether it represents an earlier or later stage.\n"
            "4. Estimate the progress numerically as a floating-point value between 0% and 100%.\n\n"
            "Your response must strictly follow this format:\n"
            "<ref_think>Your reasoning for choosing the most similar text_demo step(s) as the reference</ref_think>\n"
            "<ref>which text demo is most semantically similar to the current state, and output only the number of that text demo</ref>\n"
            "<score_think>Your reasoning for comparing the current state image with the reference step(s)</score_think>\n"
            "<score>Your final estimated progress score here</score>"
        )
        content.append({"type": "text", "text": task_instructions})
        
        return content
    
    def get_sample_unique_id(self, sample: Dict) -> str:
        """
        Generate unique identifier for sample
        
        Args:
            sample: A sample from JSONL
        
        Returns:
            Unique identifier string: id_progress_score
        """
        sample_id = sample.get('id', 'unknown')
        progress_score = sample.get('progress_score', 'unknown')
        return f"{sample_id}_{progress_score}"
    
    def load_processed_ids(self, output_file: Path) -> set:
        """
        Load processed sample IDs from output file
        
        Args:
            output_file: Output file path
        
        Returns:
            Set of processed unique IDs
        """
        processed_ids = set()
        
        if not output_file.exists():
            return processed_ids
        
        try:
            with open(output_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        result = json.loads(line.strip())
                        if 'meta_data' in result:
                            sample_id = result['meta_data'].get('id', 'unknown')
                            progress_score = result.get('ground_truth_score', 'unknown')
                            unique_id = f"{sample_id}_{progress_score}"
                            processed_ids.add(unique_id)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            print(f"⚠️  Error reading processed file: {str(e)}")
        
        return processed_ids
    
    def extract_tags(self, response: str) -> Dict[str, str]:
        """
        Extract specific tag contents from response
        
        Args:
            response: GPT-5 response text
        
        Returns:
            Dictionary with extracted contents
        """
        import re
        
        extracted = {}
        
        # Extract <ref> tag content
        ref_match = re.search(r'<ref>(.*?)</ref>', response, re.DOTALL)
        extracted['ref'] = ref_match.group(1).strip() if ref_match else None
        
        # Extract <score> tag content
        score_match = re.search(r'<score>(.*?)</score>', response, re.DOTALL)
        extracted['score'] = score_match.group(1).strip() if score_match else None
        
        return extracted
    
    def process_single_sample(self, sample: Dict) -> Dict:
        """
        Process single sample
        
        Args:
            sample: A sample from JSONL
        
        Returns:
            Processing result
        """
        try:
            # Build message content
            message_content = self.build_message_content(sample)
            
            # Call GPT-5 API
            api_params = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                "temperature": 1,
                "max_completion_tokens": 3000
            }
            
            response = self.client.chat.completions.create(**api_params)
            
            # Extract response
            assistant_response = response.choices[0].message.content
            
            # Extract tag contents
            extracted = self.extract_tags(assistant_response)
            
            # Build output result
            result = {
                "ref": extracted.get('ref'),
                "score": extracted.get('score'),
                "closest_idx": sample["closest_idx"],
                "ground_truth_score": sample["progress_score"],
                "response": assistant_response,
                "meta_data": {
                    "id": sample["id"],
                    "task_goal": sample["task_goal"],
                    "tokens_used": response.usage.total_tokens,
                    "model": self.model,
                    "timestamp": datetime.now().isoformat(),
                    "status": "success"
                }
            }
            
            return result
            
        except Exception as e:
            error_msg = f"Processing failed: {str(e)}"
            if hasattr(e, 'response'):
                error_msg += f"\nAPI response: {e.response}"
            
            return {
                "ref": None,
                "score": None,
                "closest_idx": sample.get("closest_idx", ""),
                "ground_truth_score": sample.get("progress_score", ""),
                "response": None,
                "meta_data": {
                    "id": sample["id"],
                    "task_goal": sample.get("task_goal", ""),
                    "error": error_msg,
                    "traceback": traceback.format_exc(),
                    "timestamp": datetime.now().isoformat(),
                    "status": "error"
                }
            }
    
    def save_result(self, result: Dict, output_file: Path):
        """
        Save single result to JSONL file (thread-safe)
        
        Args:
            result: Processing result
            output_file: Output file path
        """
        with write_lock:
            with open(output_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(result, ensure_ascii=False) + '\n')
    
    def process_batch(self, 
                     input_file: str, 
                     output_file: str,
                     max_workers: int = 5,
                     retry_failed: bool = True,
                     limit: int = None,
                     resume: bool = False):
        """
        Batch process JSONL file
        
        Args:
            input_file: Input JSONL file path
            output_file: Output JSONL file path
            max_workers: Maximum concurrent workers
            retry_failed: Whether to retry failed samples
            limit: Limit number of samples to process
            resume: Whether to enable resume from breakpoint
        """
        # Load input data
        all_samples = []
        with open(input_file, 'r', encoding='utf-8') as f:
            for line in f:
                all_samples.append(json.loads(line.strip()))
        
        print(f"📊 Total loaded samples: {len(all_samples)}")
        
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Resume: load processed samples
        processed_ids = set()
        if resume:
            processed_ids = self.load_processed_ids(output_path)
            if processed_ids:
                print(f"🔄 Resume mode: Found {len(processed_ids)} processed samples")
        else:
            # Non-resume mode, clear output file
            if output_path.exists():
                output_path.unlink()
                print(f"🗑️  Cleared existing output file")
        
        # Filter samples to process
        samples_to_process = []
        skipped_count = 0
        
        for sample in all_samples:
            unique_id = self.get_sample_unique_id(sample)
            if unique_id in processed_ids:
                skipped_count += 1
                continue
            samples_to_process.append(sample)
            
            # Check if limit reached
            if limit and len(samples_to_process) >= limit:
                break
        
        if skipped_count > 0:
            print(f"⏭️  Skipped {skipped_count} processed samples")
        
        if limit:
            print(f"🎯 Limit processing count: {limit}")
            samples_to_process = samples_to_process[:limit]
        
        samples = samples_to_process
        
        if not samples:
            print(f"✅ No new samples to process")
            return 0, 0
        
        print(f"🚀 Starting to process {len(samples)} samples (workers: {max_workers})")
        
        # Statistics
        success_count = 0
        error_count = 0
        total_tokens = 0
        failed_samples = []
        
        # Use thread pool for concurrent processing
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Submit all tasks
            future_to_sample = {
                executor.submit(self.process_single_sample, sample): sample 
                for sample in samples
            }
            
            # Show progress with tqdm
            desc = "Resume progress" if resume else "Processing progress"
            with tqdm(total=len(samples), desc=desc) as pbar:
                for future in as_completed(future_to_sample):
                    sample = future_to_sample[future]
                    
                    try:
                        result = future.result(timeout=60)  # 60 seconds timeout
                        
                        # Save result
                        self.save_result(result, output_path)
                        
                        # Update statistics
                        if result['meta_data']['status'] == 'success':
                            success_count += 1
                            total_tokens += result['meta_data'].get('tokens_used', 0)
                            pbar.set_postfix({
                                '✅': success_count,
                                '❌': error_count,
                                'tokens': total_tokens
                            })
                        else:
                            error_count += 1
                            failed_samples.append(sample)
                            pbar.set_postfix({
                                '✅': success_count,
                                '❌': error_count,
                                'tokens': total_tokens,
                                'last_error': result['meta_data'].get('error', '')[:50]
                            })
                        
                    except Exception as e:
                        error_count += 1
                        failed_samples.append(sample)
                        error_result = {
                            "ref": None,
                            "score": None,
                            "closest_idx": sample.get("closest_idx", ""),
                            "ground_truth_score": sample.get("progress_score", ""),
                            "response": None,
                            "meta_data": {
                                "id": sample.get("id", "unknown"),
                                "error": f"Timeout or exception: {str(e)}",
                                "timestamp": datetime.now().isoformat(),
                                "status": "error"
                            }
                        }
                        self.save_result(error_result, output_path)
                        pbar.set_postfix({
                            '✅': success_count,
                            '❌': error_count,
                            'timeout': True
                        })
                    
                    pbar.update(1)
        
        # Retry failed samples if needed
        if retry_failed and failed_samples:
            print(f"\n🔄 Retrying {len(failed_samples)} failed samples...")
            retry_success = 0
            
            with tqdm(total=len(failed_samples), desc="Retry progress") as pbar:
                for sample in failed_samples:
                    time.sleep(1)  # Avoid rate limiting
                    result = self.process_single_sample(sample)
                    self.save_result(result, output_path)
                    
                    if result['meta_data']['status'] == 'success':
                        retry_success += 1
                        success_count += 1
                        error_count -= 1
                        total_tokens += result['meta_data'].get('tokens_used', 0)
                    
                    pbar.update(1)
                    pbar.set_postfix({'Retry success': retry_success})
        
        # Print final statistics
        total_processed = success_count + error_count
        if resume and processed_ids:
            print(f"\n📊 Current processing statistics:")
            print(f"  🔄 Previously processed: {len(processed_ids)}")
            print(f"  ✨ Current processed: {total_processed}")
            print(f"    - ✅ Success: {success_count}")
            print(f"    - ❌ Failed: {error_count}")
            print(f"  📈 Total processed: {len(processed_ids) + total_processed}")
        else:
            print(f"\n📊 Processing complete:")
            print(f"  ✅ Success: {success_count}/{total_processed}")
            print(f"  ❌ Failed: {error_count}/{total_processed}")
        
        print(f"  💰 Tokens used: {total_tokens:,}")
        print(f"  📄 Results saved to: {output_path}")
        
        # Calculate estimated cost (based on GPT-5-mini pricing)
        input_cost = total_tokens * 0.25 / 1_000_000  # $0.25 per 1M input tokens
        output_cost = total_tokens * 2.0 / 1_000_000  # $2.00 per 1M output tokens
        estimated_cost = input_cost + output_cost
        print(f"  💵 Estimated cost: ${estimated_cost:.4f} (simplified calculation)")
        
        return success_count, error_count


def main():
    """Main function"""
    parser = argparse.ArgumentParser(
        description="GPT-5-mini Text-Based Progress Evaluation Processor"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        required=True,
        help="OpenAI API key"
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSONL file path"
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Output JSONL file path"
    )
    parser.add_argument(
        "--image-dir",
        type=str,
        required=True,
        help="Image base directory path"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5-mini",
        choices=["gpt-5", "gpt-5-mini", "gpt-5-nano"],
        help="GPT-5 model version to use"
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=5,
        help="Maximum concurrent workers (default: 5)"
    )
    parser.add_argument(
        "--no-retry",
        action="store_true",
        help="Do not retry failed samples"
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of samples to process"
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Enable resume from breakpoint"
    )
    
    args = parser.parse_args()
    
    # Validate input file
    if not Path(args.input).exists():
        print(f"❌ Input file does not exist: {args.input}")
        sys.exit(1)
    
    # Validate image directory
    if not Path(args.image_dir).exists():
        print(f"❌ Image directory does not exist: {args.image_dir}")
        sys.exit(1)
    
    # Create processor
    try:
        processor = TextDemoProgressProcessor(
            api_key=args.api_key,
            image_dir=args.image_dir,
            model=args.model
        )
    except Exception as e:
        print(f"❌ Initialization failed: {str(e)}")
        sys.exit(1)
    
    # Start processing
    print(f"\n{'='*60}")
    print(f"GPT-5 Text-Based Progress Evaluation")
    print(f"{'='*60}")
    print(f"📁 Input file: {args.input}")
    print(f"🖼️  Image directory: {args.image_dir}")
    print(f"🤖 Model: {args.model}")
    print(f"🔄 Resume: {'Yes' if args.resume else 'No'}")
    if args.limit:
        print(f"🎯 Processing limit: {args.limit} samples")
    print(f"{'='*60}\n")
    
    start_time = time.time()
    
    try:
        success_count, error_count = processor.process_batch(
            input_file=args.input,
            output_file=args.output,
            max_workers=args.max_workers,
            retry_failed=not args.no_retry,
            limit=args.limit,
            resume=args.resume
        )
        
        elapsed_time = time.time() - start_time
        print(f"\n⏱️  Total time: {elapsed_time:.2f} seconds")
        
        # Return appropriate exit code
        if error_count == 0:
            sys.exit(0)
        elif success_count > 0:
            sys.exit(1)  # Partial success
        else:
            sys.exit(2)  # All failed
            
    except KeyboardInterrupt:
        print("\n\n⚠️  User interrupted processing")
        sys.exit(130)
    except Exception as e:
        print(f"\n❌ Error during processing: {str(e)}")
        print(traceback.format_exc())
        sys.exit(1)


if __name__ == "__main__":
    main()