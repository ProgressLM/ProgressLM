#!/usr/bin/env python3
"""
Multithreaded Image Renaming Script
Renames frame_XXXX.jpg files to {camera_folder}_XXXX.jpg pattern
Includes dry-run mode, logging, and backup/undo functionality
"""

import os
import json
import argparse
import logging
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed
from threading import Lock
from tqdm import tqdm

# Configuration
SOURCE_DIR = "/home/vcj9002/jianshu/chengxuan/Data/robomind/data/3rgb"
SCRIPT_DIR = "/home/vcj9002/jianshu/chengxuan/ProgressLM/data/utils_img/sft_manage"
BACKUP_FILE = os.path.join(SCRIPT_DIR, "rename_backup.json")
LOG_FILE = os.path.join(SCRIPT_DIR, f"rename_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")

# Thread-safe counters
stats_lock = Lock()
stats = {
    'total': 0,
    'success': 0,
    'failed': 0,
    'skipped': 0
}


def setup_logging(quiet=False):
    """Setup logging configuration"""
    handlers = [logging.FileHandler(LOG_FILE, encoding='utf-8')]

    # Only add StreamHandler if not in quiet mode (to avoid interfering with tqdm)
    if not quiet:
        handlers.append(logging.StreamHandler())

    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


def find_all_image_files(source_dir):
    """
    Find all frame_*.jpg files in the directory tree
    Returns: list of tuples (full_path, camera_folder_name)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Scanning directory: {source_dir}")
    print(f"Scanning directory: {source_dir}")

    files_to_process = []
    dir_count = 0

    # Count total directories first for progress bar
    print("Counting directories...")
    total_dirs = sum(1 for _, _, _ in os.walk(source_dir))

    # Walk through the directory tree with progress bar
    with tqdm(total=total_dirs, desc="Scanning dirs", unit="dirs") as pbar:
        for root, dirs, files in os.walk(source_dir):
            dir_count += 1
            for filename in files:
                if filename.startswith('frame_') and filename.endswith('.jpg'):
                    full_path = os.path.join(root, filename)

                    # Extract camera folder name (first level under source_dir)
                    relative_path = os.path.relpath(root, source_dir)
                    camera_folder = relative_path.split(os.sep)[0]

                    files_to_process.append((full_path, camera_folder))

            pbar.update(1)
            pbar.set_postfix({'files_found': len(files_to_process)})

    logger.info(f"Found {len(files_to_process)} files to process")
    print(f"\nFound {len(files_to_process):,} files to process")
    return files_to_process


def get_new_filename(old_path, camera_folder):
    """
    Generate new filename based on camera folder and frame number
    Example: frame_0000.jpg -> camera_left_0000.jpg
    """
    old_filename = os.path.basename(old_path)

    # Extract frame number from frame_XXXX.jpg
    frame_number = old_filename.replace('frame_', '').replace('.jpg', '')

    # Create new filename
    new_filename = f"{camera_folder}_{frame_number}.jpg"

    # Get directory path
    directory = os.path.dirname(old_path)
    new_path = os.path.join(directory, new_filename)

    return new_path


def rename_single_file(file_info, dry_run=False):
    """
    Rename a single file
    Returns: tuple (success, old_path, new_path, error_msg)
    """
    old_path, camera_folder = file_info
    logger = logging.getLogger(__name__)

    try:
        new_path = get_new_filename(old_path, camera_folder)

        # Check if new file already exists
        if os.path.exists(new_path) and new_path != old_path:
            return (False, old_path, new_path, "Target file already exists")

        if not dry_run:
            os.rename(old_path, new_path)

        return (True, old_path, new_path, None)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error renaming {old_path}: {error_msg}")
        return (False, old_path, None, error_msg)


def rename_files_multithreaded(files_to_process, num_threads=8, dry_run=False):
    """
    Rename files using multithreading with progress bar
    Returns: backup_data dict mapping old_path -> new_path
    """
    logger = logging.getLogger(__name__)
    backup_data = {}

    mode = "DRY RUN" if dry_run else "RENAMING"
    logger.info(f"{mode}: Processing {len(files_to_process)} files with {num_threads} threads")

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        # Submit all tasks
        future_to_file = {
            executor.submit(rename_single_file, file_info, dry_run): file_info
            for file_info in files_to_process
        }

        # Process results with enhanced progress bar
        with tqdm(
            total=len(files_to_process),
            desc=mode,
            unit="files",
            unit_scale=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        ) as pbar:
            for future in as_completed(future_to_file):
                success, old_path, new_path, error_msg = future.result()

                with stats_lock:
                    if success:
                        stats['success'] += 1
                        if new_path:
                            backup_data[old_path] = new_path
                    else:
                        stats['failed'] += 1
                        # Use tqdm.write to avoid interfering with progress bar
                        if stats['failed'] <= 10:  # Only show first 10 errors in console
                            tqdm.write(f"Error: {os.path.basename(old_path)} -> {error_msg}")
                        logger.warning(f"Failed: {old_path} -> {error_msg}")

                    # Update progress bar with real-time stats
                    pbar.set_postfix({
                        'Success': f"{stats['success']:,}",
                        'Failed': stats['failed']
                    })

                pbar.update(1)

    return backup_data


def save_backup(backup_data, dry_run=False):
    """Save backup data to JSON file"""
    logger = logging.getLogger(__name__)

    if dry_run:
        logger.info(f"DRY RUN: Would save backup to {BACKUP_FILE}")
        return

    try:
        with open(BACKUP_FILE, 'w', encoding='utf-8') as f:
            json.dump(backup_data, f, indent=2, ensure_ascii=False)
        logger.info(f"Backup saved to {BACKUP_FILE}")
    except Exception as e:
        logger.error(f"Failed to save backup: {e}")


def load_backup():
    """Load backup data from JSON file"""
    logger = logging.getLogger(__name__)

    if not os.path.exists(BACKUP_FILE):
        logger.error(f"Backup file not found: {BACKUP_FILE}")
        return None

    try:
        with open(BACKUP_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load backup: {e}")
        return None


def undo_rename(num_threads=8):
    """Undo the renaming operation using backup file"""
    logger = logging.getLogger(__name__)
    logger.info("Starting undo operation...")
    print("Starting undo operation...")

    backup_data = load_backup()
    if not backup_data:
        logger.error("Cannot proceed with undo: no backup data")
        print("ERROR: Cannot proceed with undo: no backup data")
        return

    # Reverse the mapping: new_path -> old_path
    files_to_restore = [(new_path, old_path) for old_path, new_path in backup_data.items()]

    logger.info(f"Restoring {len(files_to_restore)} files...")
    print(f"Restoring {len(files_to_restore):,} files...")

    # Reset stats
    global stats
    stats = {'total': len(files_to_restore), 'success': 0, 'failed': 0, 'skipped': 0}

    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_file = {
            executor.submit(restore_single_file, new_path, old_path): (new_path, old_path)
            for new_path, old_path in files_to_restore
        }

        # Enhanced progress bar for undo operation
        with tqdm(
            total=len(files_to_restore),
            desc="RESTORING",
            unit="files",
            unit_scale=True,
            bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}] {postfix}'
        ) as pbar:
            for future in as_completed(future_to_file):
                success, error_msg = future.result()

                with stats_lock:
                    if success:
                        stats['success'] += 1
                    else:
                        stats['failed'] += 1
                        if stats['failed'] <= 10:
                            tqdm.write(f"Error during restore: {error_msg}")

                    # Update progress bar with real-time stats
                    pbar.set_postfix({
                        'Success': f"{stats['success']:,}",
                        'Failed': stats['failed']
                    })

                pbar.update(1)

    print_statistics()

    if stats['failed'] == 0:
        logger.info("Undo completed successfully. Removing backup file...")
        print("\nUndo completed successfully. Removing backup file...")
        os.remove(BACKUP_FILE)


def restore_single_file(new_path, old_path):
    """Restore a single file to its original name"""
    logger = logging.getLogger(__name__)

    try:
        if not os.path.exists(new_path):
            return (False, f"Source file not found: {new_path}")

        if os.path.exists(old_path):
            return (False, f"Target file already exists: {old_path}")

        os.rename(new_path, old_path)
        return (True, None)

    except Exception as e:
        error_msg = str(e)
        logger.error(f"Error restoring {new_path} -> {old_path}: {error_msg}")
        return (False, error_msg)


def print_statistics():
    """Print final statistics"""
    logger = logging.getLogger(__name__)

    print("\n" + "="*60)
    print("RENAMING STATISTICS")
    print("="*60)
    print(f"Total files:     {stats.get('total', stats['success'] + stats['failed'] + stats['skipped'])}")
    print(f"Successfully:    {stats['success']}")
    print(f"Failed:          {stats['failed']}")
    print(f"Skipped:         {stats['skipped']}")
    print("="*60)

    logger.info(f"Statistics - Total: {stats.get('total', 0)}, Success: {stats['success']}, "
                f"Failed: {stats['failed']}, Skipped: {stats['skipped']}")


def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description='Multithreaded image file renaming script',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Dry run to preview changes
  python rename_images.py --dry-run

  # Actually rename files with 16 threads
  python rename_images.py --threads 16

  # Undo the last renaming operation
  python rename_images.py --undo
        """
    )

    parser.add_argument('--dry-run', action='store_true',
                        help='Preview changes without actually renaming files')
    parser.add_argument('--threads', type=int, default=8,
                        help='Number of threads to use (default: 8)')
    parser.add_argument('--undo', action='store_true',
                        help='Undo the last renaming operation using backup file')

    args = parser.parse_args()

    # Setup logging - use quiet mode to avoid interfering with tqdm
    logger = setup_logging(quiet=True)

    # Print initial info to console
    print("="*60)
    print("Image Renaming Script Started")
    print("="*60)
    print(f"Source directory: {SOURCE_DIR}")
    print(f"Log file: {LOG_FILE}")
    print(f"Threads: {args.threads}")
    print(f"Dry run: {args.dry_run}")
    print("="*60)

    # Log to file
    logger.info("="*60)
    logger.info("Image Renaming Script Started")
    logger.info("="*60)
    logger.info(f"Source directory: {SOURCE_DIR}")
    logger.info(f"Log file: {LOG_FILE}")
    logger.info(f"Threads: {args.threads}")
    logger.info(f"Dry run: {args.dry_run}")

    # Handle undo operation
    if args.undo:
        undo_rename(num_threads=args.threads)
        return

    # Find all files to process
    files_to_process = find_all_image_files(SOURCE_DIR)

    if not files_to_process:
        logger.warning("No files found to process!")
        return

    stats['total'] = len(files_to_process)

    # Show preview of first few renames
    print("\nPreview of first 5 renames:")
    logger.info("\nPreview of first 5 renames:")
    for i, (old_path, camera_folder) in enumerate(files_to_process[:5]):
        new_path = get_new_filename(old_path, camera_folder)
        old_name = os.path.basename(old_path)
        new_name = os.path.basename(new_path)
        print(f"  {old_name} -> {new_name}")
        logger.info(f"  {old_name} -> {new_name}")

    if args.dry_run:
        print("\n*** DRY RUN MODE - No files will be modified ***\n")
        logger.info("\n*** DRY RUN MODE - No files will be modified ***\n")
    else:
        response = input(f"\nProceed with renaming {len(files_to_process):,} files? (yes/no): ")
        if response.lower() != 'yes':
            print("Operation cancelled by user")
            logger.info("Operation cancelled by user")
            return

    # Perform renaming
    backup_data = rename_files_multithreaded(
        files_to_process,
        num_threads=args.threads,
        dry_run=args.dry_run
    )

    # Save backup
    if backup_data:
        save_backup(backup_data, dry_run=args.dry_run)

    # Print statistics
    print_statistics()

    print("\n" + "="*60)
    print("Script completed")
    print("="*60)
    print(f"Check log file for details: {LOG_FILE}")

    logger.info("="*60)
    logger.info("Script completed")
    logger.info("="*60)


if __name__ == "__main__":
    main()
