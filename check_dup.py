import os
from difflib import SequenceMatcher
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from itertools import combinations
import hashlib
import logging
from tqdm import tqdm
import argparse
import time
from functools import lru_cache
import psutil
import sys
import shutil

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


@lru_cache(maxsize=1024)
def get_file_hash(filepath):
    """Calculate MD5 hash of file content for quick comparison"""
    hasher = hashlib.md5()
    try:
        with open(filepath, "rb") as f:
            buf = f.read(65536)  # Read in 64kb chunks
            while len(buf) > 0:
                hasher.update(buf)
                buf = f.read(65536)
        return hasher.hexdigest()
    except Exception as e:
        logger.error(f"Error calculating hash for {filepath}: {e}")
        return None


def compare_file_pair(file_pair):
    """Compare a pair of files for similarity"""
    file1, file2 = file_pair
    logger.debug(f"Comparing files: {file1.name} <-> {file2.name}")

    # Skip comparison if file sizes are significantly different (>10%)
    size1 = file1.stat().st_size
    size2 = file2.stat().st_size
    if abs(size1 - size2) > max(size1, size2) * 0.1:
        return None

    # First check filename similarity
    name_ratio = SequenceMatcher(None, file1.stem, file2.stem).ratio()
    if name_ratio <= 0.9:
        return None

    logger.debug(
        f"Name similarity {name_ratio:.2%} between {file1.name} and {file2.name}"
    )

    # Quick content comparison using hash
    hash1 = get_file_hash(file1)
    hash2 = get_file_hash(file2)

    if not hash1 or not hash2:
        return None

    if hash1 == hash2:
        logger.info(f"Found identical content between {file1.name} and {file2.name}")
        return file2 if len(file1.stem) <= len(file2.stem) else file1

    # If hashes are different but names are similar, do detailed content comparison
    # Only compare files smaller than 1MB with content ratio to avoid memory issues
    if max(size1, size2) < 1_000_000:
        try:
            with (
                open(file1, "r", encoding="utf-8", errors="ignore") as f1,
                open(file2, "r", encoding="utf-8", errors="ignore") as f2,
            ):
                content1 = f1.read()
                content2 = f2.read()

            content_ratio = SequenceMatcher(None, content1, content2).ratio()
            if content_ratio > 0.90:
                logger.info(
                    f"Found similar content ({content_ratio:.2%}) between {file1.name} and {file2.name}"
                )
                return file2 if len(file1.stem) <= len(file2.stem) else file1
        except Exception as e:
            logger.warning(f"Error comparing file contents: {e}")

    return None


def get_optimal_workers(total_files, batch_size):
    """Calculate optimal worker count based on system resources and dataset size"""
    # Get system information
    cpu_count = os.cpu_count() or 4
    available_memory_gb = psutil.virtual_memory().available / (1024**3)

    # Base workers on CPU count
    base_workers = max(1, cpu_count - 1)  # Leave one CPU for system

    # Adjust based on available memory (each worker might use ~100MB)
    memory_based_workers = int(available_memory_gb * 10)  # ~100MB per worker

    # Adjust based on dataset size (larger datasets need fewer workers to avoid memory issues)
    if total_files > 50000:
        size_factor = 0.3  # Reduce workers for very large datasets
    elif total_files > 20000:
        size_factor = 0.5
    elif total_files > 10000:
        size_factor = 0.7
    else:
        size_factor = 1.0

    # Calculate optimal workers
    optimal_workers = min(base_workers, memory_based_workers)
    optimal_workers = max(1, int(optimal_workers * size_factor))

    # Cap at a reasonable maximum
    optimal_workers = min(32, optimal_workers)

    return optimal_workers


def check_file_similarity(
    directory_path,
    max_workers=None,
    file_pattern="*.txt",
    batch_size=1000,
    auto_workers=False,
):
    """
    Optimized version to check for similar files in the given directory
    and remove duplicates based on filename and content similarity
    """
    start_time = time.time()

    # Convert to absolute path and verify directory exists
    abs_path = Path(directory_path).resolve()
    logger.info(f"Checking directory: {abs_path}")

    if not abs_path.exists():
        logger.error(f"Directory does not exist: {abs_path}")
        return
    if not abs_path.is_dir():
        logger.error(f"Path is not a directory: {abs_path}")
        return

    # Get all matching files in directory with progress feedback
    logger.info(f"Scanning for files matching '{file_pattern}'...")
    txt_files = list(abs_path.glob(file_pattern))
    total_files = len(txt_files)
    logger.info(f"Found {total_files} files matching '{file_pattern}' to process")

    if not txt_files:
        logger.warning(f"No {file_pattern} files found in directory!")
        return

    # Process files in batches to avoid memory issues with large datasets
    if total_files > 10000:
        logger.warning(
            f"Large number of files detected ({total_files}). Processing in batches."
        )

    # Set optimal worker count if auto_workers is enabled
    if auto_workers or max_workers is None:
        max_workers = get_optimal_workers(total_files, batch_size)
        logger.info(
            f"Auto-configured to use {max_workers} workers based on system resources and dataset size"
        )
    else:
        max_workers = max_workers or min(32, os.cpu_count() + 4)  # Limit max workers

    files_to_remove = set()

    # Calculate total number of pairs to give context
    total_pairs = (total_files * (total_files - 1)) // 2
    logger.info(f"Will process approximately {total_pairs} file pairs in batches")

    # Process in smaller batches to prevent memory issues
    processed_pairs = 0
    batch_start_time = time.time()
    overall_start_time = time.time()

    # Track progress for better time estimation
    batch_times = []
    remaining_batches = (total_files + batch_size - 1) // batch_size

    # Function to process a batch of files
    def process_batch(start_idx, end_idx, batch_num):
        nonlocal processed_pairs
        batch_files = txt_files[start_idx:end_idx]

        # Generate comparisons between files in this batch and all previous files
        current_batch_pairs = []
        for i, file1 in enumerate(batch_files):
            # Within batch comparisons
            for file2 in batch_files[i + 1 :]:
                current_batch_pairs.append((file1, file2))

            # Comparisons with files from previous batches
            if start_idx > 0:
                for file2 in txt_files[:start_idx]:
                    current_batch_pairs.append((file1, file2))

        logger.info(
            f"Processing batch {batch_num}/{remaining_batches} with {len(current_batch_pairs)} file pairs..."
        )
        batch_to_remove = set()

        chunk_size = max(1, len(current_batch_pairs) // max_workers // 10)
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            with tqdm(
                total=len(current_batch_pairs),
                desc=f"Batch {batch_num}/{remaining_batches}",
                unit="pairs",
                ncols=100,
            ) as pbar:
                for result in executor.map(
                    compare_file_pair, current_batch_pairs, chunksize=chunk_size
                ):
                    if result:
                        batch_to_remove.add(result)
                    pbar.update(1)

        processed_pairs += len(current_batch_pairs)
        percent_complete = processed_pairs / total_pairs if total_pairs > 0 else 0
        logger.info(
            f"Batch complete: {processed_pairs}/{total_pairs} pairs processed ({percent_complete:.1%})"
        )
        return batch_to_remove

    # Process files in batches
    for i in range(0, total_files, batch_size):
        batch_num = (i // batch_size) + 1
        batch_end = min(i + batch_size, total_files)

        batch_start = time.time()
        batch_files_to_remove = process_batch(i, batch_end, batch_num)
        batch_duration = time.time() - batch_start
        batch_times.append(batch_duration)

        files_to_remove.update(batch_files_to_remove)

        # Calculate time estimates
        elapsed_total = time.time() - overall_start_time
        avg_batch_time = sum(batch_times) / len(batch_times)
        remaining_batches = (total_files - batch_end) // batch_size + (
            1 if (total_files - batch_end) % batch_size > 0 else 0
        )
        estimated_remaining = avg_batch_time * remaining_batches

        # Calculate estimated completion time
        eta = time.strftime(
            "%H:%M:%S", time.localtime(time.time() + estimated_remaining)
        )

        # Log batch completion with detailed timing information
        logger.info(
            f"Batch {batch_num} completed in {batch_duration:.1f}s | "
            f"Elapsed: {elapsed_total:.1f}s | "
            f"Remaining: ~{estimated_remaining:.1f}s | "
            f"ETA: {eta} | "
            f"Avg. batch time: {avg_batch_time:.1f}s"
        )

    logger.info(f"Found {len(files_to_remove)} duplicate files to remove")

    # Remove duplicate files with progress bar
    if files_to_remove:
        with tqdm(
            total=len(files_to_remove), desc="Removing duplicates", unit="files"
        ) as pbar:
            for file_to_remove in files_to_remove:
                try:
                    os.remove(file_to_remove)
                    logger.info(
                        f"Successfully removed duplicate file: {file_to_remove}"
                    )
                except Exception as e:
                    logger.error(f"Error removing file {file_to_remove}: {e}")
                pbar.update(1)

    elapsed_time = time.time() - start_time
    logger.info(f"File similarity check completed in {elapsed_time:.2f} seconds")


def handle_drag_and_drop():
    """Handle a file or folder path that was dropped onto the script"""
    if len(sys.argv) == 2 and os.path.exists(sys.argv[1]) and sys.argv[1] != "--help":
        path = Path(sys.argv[1])

        if path.is_file():
            # If a file was dropped, copy it to the default directory and process that directory
            target_dir = Path(__file__).parent / "data" / "txts"
            target_dir.mkdir(parents=True, exist_ok=True)

            try:
                target_file = target_dir / path.name
                shutil.copy2(path, target_file)
                logger.info(f"Copied file {path.name} to {target_dir}")
                return target_dir
            except Exception as e:
                logger.error(f"Error copying file: {e}")
                return None
        elif path.is_dir():
            # If a directory was dropped, process that directory
            return path

    return None


def process_uploads(source_path, destination_path):
    """
    Copy files from source to destination before processing
    """
    destination = Path(destination_path)
    destination.mkdir(parents=True, exist_ok=True)

    source = Path(source_path)
    if source.is_file():
        # Copy single file
        try:
            target_file = destination / source.name
            shutil.copy2(source, target_file)
            logger.info(f"Uploaded file {source.name} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error uploading file: {e}")
            return False
    elif source.is_dir():
        # Copy all matching files
        try:
            file_count = 0
            for file in source.glob("*"):
                if file.is_file():
                    target_file = destination / file.name
                    shutil.copy2(file, target_file)
                    file_count += 1

            logger.info(f"Uploaded {file_count} files from {source} to {destination}")
            return True
        except Exception as e:
            logger.error(f"Error uploading files: {e}")
            return False

    return False


if __name__ == "__main__":
    # Handle drag and drop first
    drag_drop_path = handle_drag_and_drop()

    if drag_drop_path:
        logger.info(f"Processing drag-and-dropped path: {drag_drop_path}")
        check_file_similarity(
            drag_drop_path,
            max_workers=None,
            file_pattern="*.*",  # Process all files by default for drag-drop
            batch_size=1000,
            auto_workers=True,
        )
    else:
        # Regular command-line argument handling
        parser = argparse.ArgumentParser(
            description="Check for and remove duplicate files"
        )
        parser.add_argument("--dir", "-d", type=str, help="Directory to check files in")
        parser.add_argument(
            "--pattern",
            "-p",
            type=str,
            default="*.txt",
            help="File pattern to match (default: *.txt)",
        )
        parser.add_argument(
            "--workers",
            "-w",
            type=int,
            default=None,
            help="Number of worker processes (default: auto)",
        )
        parser.add_argument(
            "--batch",
            "-b",
            type=int,
            default=1000,
            help="Batch size for processing files (default: 1000)",
        )
        parser.add_argument(
            "--auto",
            "-a",
            action="store_true",
            help="Auto-adjust worker count based on system resources and dataset size",
        )
        parser.add_argument(
            "--upload",
            "-u",
            type=str,
            help="Upload a file or folder to the processing directory",
        )
        parser.add_argument(
            "--target",
            "-t",
            type=str,
            help="Target directory for uploaded files (default: data/txts)",
        )
        args = parser.parse_args()

        # Handle file/folder uploads if specified
        if args.upload:
            upload_source = Path(args.upload)
            upload_target = (
                Path(args.target)
                if args.target
                else Path(__file__).parent / "data" / "txts"
            )

            if process_uploads(upload_source, upload_target):
                directory = upload_target
                logger.info(f"Will process uploaded files in {directory}")
            else:
                logger.error("Failed to process uploads. Exiting.")
                sys.exit(1)
        elif args.dir:
            directory = Path(args.dir)
        else:
            # Use default path for testing
            current_dir = Path(__file__).parent
            directory = current_dir / "data" / "txts"
            logger.info(f"No directory specified. Using default: {directory}")

        check_file_similarity(
            directory,
            max_workers=args.workers,
            file_pattern=args.pattern,
            batch_size=args.batch,
            auto_workers=args.auto,
        )
