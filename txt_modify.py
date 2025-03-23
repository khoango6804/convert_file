import os
import shutil
import argparse
import logging
import sys
import traceback
import multiprocessing
from concurrent.futures import ProcessPoolExecutor
import time
from functools import partial
import json
import hashlib

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)


def get_checkpoint_file(target_dir):
    """Get the path to the checkpoint file."""
    return os.path.join(target_dir, ".consolidate_checkpoint.json")


def load_checkpoint(checkpoint_file):
    """Load processed files from checkpoint file."""
    if os.path.exists(checkpoint_file):
        try:
            with open(checkpoint_file, "r") as f:
                return json.load(f)
        except Exception as e:
            logger.warning(f"Could not load checkpoint file: {e}")
    return {"processed_files": []}


def save_checkpoint(checkpoint_file, processed_files):
    """Save processed files to checkpoint file."""
    try:
        checkpoint_data = {"processed_files": processed_files}
        with open(checkpoint_file, "w") as f:
            json.dump(checkpoint_data, f)
    except Exception as e:
        logger.error(f"Failed to save checkpoint: {e}")


def get_file_identifier(root, file):
    """Generate a unique identifier for a file based on path and modification time."""
    source_path = os.path.join(root, file)
    try:
        mtime = os.path.getmtime(source_path)
        size = os.path.getsize(source_path)
        return hashlib.md5(f"{source_path}_{mtime}_{size}".encode()).hexdigest()
    except Exception:
        # If we can't get mtime or size, just use the path
        return hashlib.md5(source_path.encode()).hexdigest()


def process_file(
    file_info, target_dir, processed_files, method="copy", overwrite=False
):
    """Process a single file - for multiprocessing use"""
    root, file, file_id = file_info

    # Skip if already processed
    if file_id in processed_files:
        return (True, source_path, target_path, file_id, "skipped")

    source_path = os.path.join(root, file)
    target_path = os.path.join(target_dir, file)

    # Check if file already exists in target directory
    if os.path.exists(target_path) and not overwrite:
        base, ext = os.path.splitext(file)
        counter = 1
        while os.path.exists(target_path):
            new_name = f"{base}_{counter}{ext}"
            target_path = os.path.join(target_dir, new_name)
            counter += 1

    try:
        # Copy or move the file
        if method == "copy":
            shutil.copy2(source_path, target_path)
        else:  # 'move'
            shutil.move(source_path, target_path)
        return (True, source_path, target_path, file_id, "processed")
    except Exception as e:
        return (False, source_path, str(e), file_id, "failed")


def consolidate_files(
    source_dir,
    target_dir,
    method="copy",
    overwrite=False,
    max_workers=None,
    checkpoint=True,
    batch_size=100,
):
    """
    Consolidate all files from source_dir (including subdirectories) into a single target_dir.

    Parameters:
    - source_dir: Directory containing files to consolidate (including subdirectories)
    - target_dir: Target directory where all files will be placed
    - method: 'copy' or 'move' (default: 'copy')
    - overwrite: Whether to overwrite files in target directory if they already exist (default: False)
    - max_workers: Maximum number of worker processes (default: CPU count)
    - checkpoint: Whether to use checkpointing (default: True)
    - batch_size: Number of files to process before saving checkpoint (default: 100)

    Returns:
    - Dictionary with counts of processed, skipped, and failed files
    """
    try:
        start_time = time.time()

        # Verify the source directory exists
        if not os.path.exists(source_dir):
            logger.error(f"Error: Source directory '{source_dir}' does not exist!")
            return {"processed": 0, "skipped": 0, "failed": 0, "time": 0}

        # Create target directory if it doesn't exist
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
            logger.info(f"Created target directory: {target_dir}")

        # Load checkpoint if enabled
        checkpoint_file = get_checkpoint_file(target_dir)
        processed_files = []
        if checkpoint:
            checkpoint_data = load_checkpoint(checkpoint_file)
            processed_files = checkpoint_data.get("processed_files", [])
            if processed_files:
                logger.info(
                    f"Loaded {len(processed_files)} processed files from checkpoint"
                )

        stats = {"processed": 0, "skipped": 0, "failed": 0}

        # Get all files first - faster than repeatedly walking the directory
        file_list = []
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_id = get_file_identifier(root, file)
                file_list.append((root, file, file_id))

        if not file_list:
            logger.warning(f"No files found in source directory: {source_dir}")
            return {"processed": 0, "skipped": 0, "failed": 0, "time": 0}

        total_files = len(file_list)
        logger.info(f"Found {total_files} files to process")

        # Filter out already processed files if using checkpoint
        if checkpoint and processed_files:
            initial_count = len(file_list)
            file_list = [item for item in file_list if item[2] not in processed_files]
            skipped_count = initial_count - len(file_list)
            stats["skipped"] = skipped_count
            logger.info(f"Skipping {skipped_count} already processed files")

        if not file_list:
            logger.info("All files have already been processed!")
            elapsed_time = time.time() - start_time
            stats["time"] = elapsed_time
            return stats

        # Determine optimal number of workers
        if max_workers is None:
            max_workers = 12

        # Process files in parallel using a process pool
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            process_func = partial(
                process_file,
                target_dir=target_dir,
                processed_files=processed_files,
                method=method,
                overwrite=overwrite,
            )

            # Process in batches for checkpointing
            newly_processed = []
            batch_count = 0

            for i in range(0, len(file_list), batch_size):
                batch = file_list[i : i + batch_size]
                batch_results = list(executor.map(process_func, batch))

                # Process batch results
                for result in batch_results:
                    success, source_path, info, file_id, status = result

                    if status == "skipped":
                        stats["skipped"] += 1
                    elif success:
                        stats["processed"] += 1
                        newly_processed.append(file_id)
                    else:
                        stats["failed"] += 1
                        logger.error(f"Error processing {source_path}: {info}")

                # Update progress
                batch_count += 1
                current_count = stats["processed"] + stats["skipped"]
                logger.info(
                    f"Progress: {current_count}/{total_files} files processed (batch {batch_count})"
                )

                # Save checkpoint after each batch if enabled
                if checkpoint and newly_processed:
                    save_checkpoint(checkpoint_file, processed_files + newly_processed)
                    logger.debug(f"Checkpoint saved after batch {batch_count}")

        # Final checkpoint save
        if checkpoint and newly_processed:
            save_checkpoint(checkpoint_file, processed_files + newly_processed)
            logger.info("Final checkpoint saved")

        elapsed_time = time.time() - start_time
        stats["time"] = elapsed_time
        return stats

    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}")
        logger.error(traceback.format_exc())
        return {"processed": 0, "skipped": 0, "failed": 0, "time": 0}


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(
            description="Consolidate files from multiple folders into one folder"
        )
        parser.add_argument(
            "--source",
            default=r"C:\Users\lunovian\Documents\Github\convert_file\data\full_txt",
            help="Source directory containing files (default: data folder)",
        )
        parser.add_argument(
            "--target",
            default=r"C:\Users\lunovian\Documents\Github\convert_file\data\final_data",
            help="Target directory where all files will be placed",
        )
        parser.add_argument(
            "--method",
            choices=["copy", "move"],
            default="copy",
            help="Method to use: copy or move files (default: copy)",
        )
        parser.add_argument(
            "--overwrite",
            action="store_true",
            help="Overwrite files in target directory if they already exist",
        )
        parser.add_argument(
            "--debug",
            action="store_true",
            help="Enable debug mode with more detailed logging",
        )
        parser.add_argument(
            "--workers",
            type=int,
            default=None,
            help="Number of worker processes for parallel execution (default: CPU count)",
        )
        parser.add_argument(
            "--no-checkpoint",
            action="store_true",
            help="Disable checkpointing (don't skip already processed files)",
        )
        parser.add_argument(
            "--batch-size",
            type=int,
            default=100,
            help="Number of files to process in a batch before saving checkpoint (default: 100)",
        )

        args = parser.parse_args()

        if args.debug:
            logger.setLevel(logging.DEBUG)

        logger.info(f"Starting file consolidation from {args.source} to {args.target}")
        logger.info(f"Method: {args.method.upper()}, Overwrite: {args.overwrite}")
        logger.info(
            f"Starting file consolidation with {args.workers or 'auto'} workers"
        )
        start_time = time.time()

        stats = consolidate_files(
            args.source,
            args.target,
            args.method,
            args.overwrite,
            args.workers,
            checkpoint=not args.no_checkpoint,
            batch_size=args.batch_size,
        )

        logger.info("\nSummary:")
        logger.info(f"- Files processed: {stats['processed']}")
        logger.info(f"- Files skipped: {stats['skipped']}")
        logger.info(f"- Files failed: {stats['failed']}")
        logger.info(f"- Time taken: {stats['time']:.2f} seconds")

        files_per_second = (
            stats["processed"] / stats["time"] if stats["time"] > 0 else 0
        )
        logger.info(f"- Performance: {files_per_second:.1f} files/second")
        logger.info("Done!")
    except Exception as e:
        logger.error(f"Script execution failed: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)
