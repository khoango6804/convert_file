import os
import hashlib
import time
import threading
import concurrent.futures
import argparse
import json
import pickle
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import logging
from datetime import datetime


class DuplicateFileFinder:
    def __init__(
        self,
        directory,
        recursive=True,
        exclude_extensions=None,
        include_extensions=None,
        max_depth=None,
        chunk_size=8192,
        num_threads=os.cpu_count(),
        keep_strategy="newest",
        resume=False,
    ):
        self.directory = directory
        self.recursive = recursive
        self.exclude_extensions = exclude_extensions or []
        self.include_extensions = include_extensions or []
        self.max_depth = max_depth
        self.chunk_size = chunk_size
        self.num_threads = num_threads
        self.keep_strategy = keep_strategy  # 'newest', 'oldest', 'first_found'
        self.resume = resume

        # Checkpoint file paths
        self.checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints"
        )
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        self.checkpoint_file = os.path.join(
            self.checkpoint_dir, f"dedup_checkpoint_{Path(self.directory).name}.pickle"
        )

        # Logging setup
        self.setup_logging()

        # Results storage
        self.size_groups = defaultdict(list)
        self.filename_duplicates = []
        self.content_duplicates = []
        self.current_stage = "initialized"  # Tracks the current processing stage

        # Statistics
        self.stats = {
            "total_files": 0,
            "total_size": 0,
            "duplicate_files": 0,
            "duplicate_size": 0,
            "time_taken": 0,
        }

        # Try to resume from checkpoint if requested
        if self.resume:
            self._load_checkpoint()

    def setup_logging(self):
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(
            log_dir, f"dedup_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )
        self.logger = logging.getLogger(__name__)

    def scan_directory(self):
        """Scan directory and group files by size"""
        if self.current_stage != "initialized" and self.resume:
            self.logger.info(f"Resuming from stage: {self.current_stage}")
            return self.size_groups

        self.current_stage = "scanning"
        start_time = time.time()
        self.logger.info(f"Starting scan of directory: {self.directory}")

        all_files = []

        # Walk through directories and collect files
        for root, _, files in os.walk(self.directory):
            # Check depth constraint
            if self.max_depth is not None:
                relative_path = os.path.relpath(root, self.directory)
                if (
                    relative_path != "."
                    and relative_path.count(os.sep) >= self.max_depth
                ):
                    continue

            for file in files:
                file_path = os.path.join(root, file)

                # Skip if not a file
                if not os.path.isfile(file_path):
                    continue

                # Check extension filters
                ext = os.path.splitext(file)[1].lower()
                if (self.include_extensions and ext not in self.include_extensions) or (
                    ext in self.exclude_extensions
                ):
                    continue

                all_files.append(file_path)

        self.stats["total_files"] = len(all_files)
        self.logger.info(f"Found {self.stats['total_files']} files to process")

        # Group files by size with progress bar
        with tqdm(total=len(all_files), desc="Grouping files by size") as pbar:
            for file_path in all_files:
                try:
                    file_size = os.path.getsize(file_path)
                    self.size_groups[file_size].append(file_path)
                    self.stats["total_size"] += file_size
                except (OSError, IOError) as e:
                    self.logger.error(f"Error accessing file {file_path}: {e}")
                pbar.update(1)

        # Remove groups with only one file (can't have duplicates)
        self.size_groups = {
            size: files for size, files in self.size_groups.items() if len(files) > 1
        }

        elapsed = time.time() - start_time
        self.logger.info(f"Completed size grouping in {elapsed:.2f} seconds")
        self.logger.info(
            f"Found {len(self.size_groups)} size groups with potential duplicates"
        )

        # Save checkpoint after scanning
        self._save_checkpoint()

        return self.size_groups

    def compare_filenames(self):
        """Compare filenames within each size group"""
        if self.current_stage == "filename_compared" and self.resume:
            self.logger.info("Skipping filename comparison (already completed)")
            return self.filename_duplicates

        self.current_stage = "comparing_filenames"
        self.logger.info("Starting filename comparison")
        start_time = time.time()

        # Process each size group
        total_groups = len(self.size_groups)
        with tqdm(total=total_groups, desc="Comparing filenames") as pbar:
            for size, files in self.size_groups.items():
                # Group by filename
                name_groups = defaultdict(list)
                for file_path in files:
                    filename = os.path.basename(file_path)
                    name_groups[filename].append(file_path)

                # Find duplicate filenames
                for name, file_paths in name_groups.items():
                    if len(file_paths) > 1:
                        # These are duplicates by filename
                        sorted_files = sorted(file_paths, key=os.path.getmtime)
                        if self.keep_strategy == "newest":
                            to_keep = sorted_files[-1]
                            to_delete = sorted_files[:-1]
                        elif self.keep_strategy == "oldest":
                            to_keep = sorted_files[0]
                            to_delete = sorted_files[1:]
                        else:  # 'first_found'
                            to_keep = file_paths[0]
                            to_delete = file_paths[1:]

                        self.filename_duplicates.append(
                            {"keep": to_keep, "delete": to_delete, "size": size}
                        )

                        # Update statistics
                        self.stats["duplicate_files"] += len(to_delete)
                        self.stats["duplicate_size"] += size * len(to_delete)

                        # Remove these files from further comparisons
                        for file_path in file_paths:
                            files.remove(file_path)

                # If after removing filename duplicates we still have multiple files,
                # keep this group for content comparison
                if len(files) > 1:
                    self.size_groups[size] = files
                else:
                    # No need for content comparison if only 0 or 1 files left
                    self.size_groups.pop(size)

                pbar.update(1)

        elapsed = time.time() - start_time
        self.logger.info(f"Completed filename comparison in {elapsed:.2f} seconds")
        self.logger.info(
            f"Found {len(self.filename_duplicates)} groups of filename duplicates"
        )
        self.logger.info(
            f"{len(self.size_groups)} size groups remain for content comparison"
        )

        self.current_stage = "filename_compared"
        # Save checkpoint after filename comparison
        self._save_checkpoint()

        return self.filename_duplicates

    def calculate_file_hash(self, file_path, first_chunk_only=False):
        """Calculate MD5 hash of file content, optionally only first chunk"""
        hash_md5 = hashlib.md5()

        try:
            with open(file_path, "rb") as f:
                if first_chunk_only:
                    chunk = f.read(self.chunk_size)
                    hash_md5.update(chunk)
                else:
                    while chunk := f.read(self.chunk_size):
                        hash_md5.update(chunk)
            return hash_md5.hexdigest()
        except (IOError, OSError) as e:
            self.logger.error(f"Error hashing file {file_path}: {e}")
            return None

    def compare_file_content(self):
        """Compare file content within remaining size groups"""
        if self.current_stage == "content_compared" and self.resume:
            self.logger.info("Skipping content comparison (already completed)")
            return self.content_duplicates

        self.current_stage = "comparing_content"
        self.logger.info("Starting content comparison")
        start_time = time.time()

        # First pass: Check first chunk hash to quickly filter out files
        for size, files in self.size_groups.items():
            if len(files) <= 1:
                continue

            # Calculate first chunk hashes with progress bar
            with tqdm(
                total=len(files), desc=f"Quick hash check (size: {size} bytes)"
            ) as pbar:
                first_chunk_hashes = defaultdict(list)

                def process_file(file_path):
                    first_chunk_hash = self.calculate_file_hash(
                        file_path, first_chunk_only=True
                    )
                    return file_path, first_chunk_hash

                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.num_threads
                ) as executor:
                    for file_path, file_hash in executor.map(process_file, files):
                        if file_hash:
                            first_chunk_hashes[file_hash].append(file_path)
                        pbar.update(1)

                # For groups with the same first chunk, do full hash
                potential_duplicates = []
                for hash_val, file_list in first_chunk_hashes.items():
                    if len(file_list) > 1:
                        potential_duplicates.extend(file_list)

                self.size_groups[size] = potential_duplicates

        # Second pass: Full file hash for remaining potential duplicates
        for size, files in list(self.size_groups.items()):
            if len(files) <= 1:
                self.size_groups.pop(size)
                continue

            with tqdm(
                total=len(files), desc=f"Full content hash (size: {size} bytes)"
            ) as pbar:
                content_hashes = defaultdict(list)

                def process_full_file(file_path):
                    full_hash = self.calculate_file_hash(file_path)
                    return file_path, full_hash

                # Use ThreadPoolExecutor for parallel processing
                with concurrent.futures.ThreadPoolExecutor(
                    max_workers=self.num_threads
                ) as executor:
                    for file_path, file_hash in executor.map(process_full_file, files):
                        if file_hash:
                            content_hashes[file_hash].append(file_path)
                        pbar.update(1)

                # Process duplicate content files
                for hash_val, file_list in content_hashes.items():
                    if len(file_list) > 1:
                        sorted_files = sorted(file_list, key=os.path.getmtime)
                        if self.keep_strategy == "newest":
                            to_keep = sorted_files[-1]
                            to_delete = sorted_files[:-1]
                        elif self.keep_strategy == "oldest":
                            to_keep = sorted_files[0]
                            to_delete = sorted_files[1:]
                        else:  # 'first_found'
                            to_keep = file_list[0]
                            to_delete = file_list[1:]

                        self.content_duplicates.append(
                            {
                                "keep": to_keep,
                                "delete": to_delete,
                                "size": size,
                                "hash": hash_val,
                            }
                        )

                        # Update statistics
                        self.stats["duplicate_files"] += len(to_delete)
                        self.stats["duplicate_size"] += size * len(to_delete)

            # Remove this size group as it's been processed
            self.size_groups.pop(size)

        elapsed = time.time() - start_time
        self.logger.info(f"Completed content comparison in {elapsed:.2f} seconds")
        self.logger.info(
            f"Found {len(self.content_duplicates)} groups of content duplicates"
        )

        self.current_stage = "content_compared"
        # Save checkpoint after content comparison
        self._save_checkpoint()

        return self.content_duplicates

    def process_duplicates(self, action="report", symlink=False):
        """Process duplicates according to specified action"""
        all_duplicates = self.filename_duplicates + self.content_duplicates

        if not all_duplicates:
            self.logger.info("No duplicates found")
            return

        self.logger.info(
            f"Processing {len(all_duplicates)} duplicate groups with action: {action}"
        )

        if action == "report":
            self._generate_report(all_duplicates)
        elif action == "delete":
            self._delete_duplicates(all_duplicates, symlink)
        elif action == "interactive":
            self._interactive_delete(all_duplicates, symlink)

    def _generate_report(self, duplicate_groups):
        """Generate a detailed report of duplicates"""
        report_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "reports"
        )
        os.makedirs(report_path, exist_ok=True)
        report_file = os.path.join(
            report_path,
            f"duplicate_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
        )

        try:
            # Explicitly use UTF-8 encoding with error handling
            with open(report_file, "w", encoding="utf-8", errors="replace") as f:
                f.write(
                    f"Duplicate File Report - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                )
                f.write(f"Scanned directory: {self.directory}\n")
                f.write(f"Total files scanned: {self.stats['total_files']}\n")
                f.write(
                    f"Total size of all files: {self._format_size(self.stats['total_size'])}\n"
                )
                f.write(f"Duplicate files found: {self.stats['duplicate_files']}\n")
                f.write(
                    f"Duplicate files size: {self._format_size(self.stats['duplicate_size'])}\n\n"
                )

                for i, group in enumerate(duplicate_groups, 1):
                    f.write(
                        f"Group {i}: {len(group['delete']) + 1} files of size {self._format_size(group['size'])}\n"
                    )
                    try:
                        f.write(f"  Keep: {group['keep']}\n")
                        for del_file in group["delete"]:
                            f.write(f"  Delete: {del_file}\n")
                    except Exception as e:
                        # Handle any encoding issues with specific file paths
                        self.logger.warning(
                            f"Error writing file path in report, group {i}: {e}"
                        )
                        f.write(f"  Keep: [Path encoding error - see log]\n")
                        f.write(f"  Delete: [One or more paths with encoding issues]\n")
                    f.write("\n")

            self.logger.info(f"Report generated at {report_file}")
            return report_file
        except Exception as e:
            self.logger.error(f"Error generating report: {e}")
            self.logger.info(
                "Attempting to create simplified report with path encoding issues fixed"
            )

            # Fallback to a simplified report if the full one fails
            simple_report_file = os.path.join(
                report_path,
                f"simplified_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
            )

            try:
                with open(
                    simple_report_file, "w", encoding="utf-8", errors="replace"
                ) as f:
                    f.write(
                        f"Simplified Duplicate File Report (encoding issues detected)\n"
                    )
                    f.write(
                        f"Total duplicates found: {self.stats['duplicate_files']} files\n"
                    )
                    f.write(
                        f"Total duplicate size: {self._format_size(self.stats['duplicate_size'])}\n\n"
                    )

                    for i, group in enumerate(duplicate_groups, 1):
                        f.write(
                            f"Group {i}: {len(group['delete']) + 1} files of size {self._format_size(group['size'])}\n\n"
                        )

                self.logger.info(f"Simplified report generated at {simple_report_file}")
                return simple_report_file
            except Exception as e2:
                self.logger.error(f"Failed to generate even simplified report: {e2}")
                return None

    def _delete_duplicates(self, duplicate_groups, create_symlinks=False):
        """Delete duplicate files, optionally creating symlinks"""
        delete_count = 0
        error_count = 0

        with tqdm(
            total=self.stats["duplicate_files"], desc="Deleting duplicates"
        ) as pbar:
            for group in duplicate_groups:
                keep_file = group["keep"]

                for delete_file in group["delete"]:
                    try:
                        os.remove(delete_file)

                        if create_symlinks:
                            # Create symbolic link pointing to the kept file
                            try:
                                # Need to get relative path if files are in different directories
                                target_dir = os.path.dirname(delete_file)
                                relative_path = os.path.relpath(keep_file, target_dir)
                                os.symlink(relative_path, delete_file)
                                self.logger.info(
                                    f"Created symlink {delete_file} -> {relative_path}"
                                )
                            except OSError as e:
                                self.logger.error(
                                    f"Error creating symlink for {delete_file}: {e}"
                                )
                                error_count += 1

                        delete_count += 1
                    except OSError as e:
                        self.logger.error(f"Error deleting file {delete_file}: {e}")
                        error_count += 1
                    pbar.update(1)

        self.logger.info(f"Deleted {delete_count} duplicate files")
        if error_count > 0:
            self.logger.warning(f"Encountered {error_count} errors during deletion")

    def _interactive_delete(self, duplicate_groups, create_symlinks=False):
        """Interactive deletion interface"""
        self.logger.info("Starting interactive duplicate management")

        for i, group in enumerate(duplicate_groups, 1):
            print(
                f"\nGroup {i}/{len(duplicate_groups)}: {len(group['delete']) + 1} files of size {self._format_size(group['size'])}"
            )
            print(f"Automatically selected to keep: {group['keep']}")

            # List files to delete
            for j, del_file in enumerate(group["delete"], 1):
                print(f"  {j}. {del_file}")

            while True:
                choice = input(
                    "\nOptions:\n"
                    "  d = delete all duplicates\n"
                    "  k = keep all (skip this group)\n"
                    "  c = change file to keep\n"
                    "  s = create symlinks instead of deleting\n"
                    "  q = quit interactive mode\n"
                    "Your choice? "
                ).lower()

                if choice == "d":
                    for del_file in group["delete"]:
                        try:
                            os.remove(del_file)
                            self.logger.info(f"Deleted {del_file}")
                        except OSError as e:
                            self.logger.error(f"Error deleting {del_file}: {e}")
                    break

                elif choice == "k":
                    self.logger.info(f"Skipped group {i}")
                    break

                elif choice == "c":
                    files = [group["keep"]] + group["delete"]
                    for j, file in enumerate(files, 1):
                        print(f"  {j}. {file}")

                    try:
                        keep_idx = int(input("Enter number of file to keep: ")) - 1
                        if 0 <= keep_idx < len(files):
                            new_keep = files[keep_idx]
                            files.pop(keep_idx)
                            group["keep"] = new_keep
                            group["delete"] = files
                            print(f"Will now keep {new_keep}")
                        else:
                            print("Invalid selection")
                    except ValueError:
                        print("Please enter a number")

                elif choice == "s":
                    for del_file in group["delete"]:
                        try:
                            os.remove(del_file)
                            target_dir = os.path.dirname(del_file)
                            relative_path = os.path.relpath(group["keep"], target_dir)
                            os.symlink(relative_path, del_file)
                            self.logger.info(
                                f"Created symlink {del_file} -> {relative_path}"
                            )
                        except OSError as e:
                            self.logger.error(f"Error processing {del_file}: {e}")
                    break

                elif choice == "q":
                    self.logger.info("Exiting interactive mode")
                    return

                else:
                    print("Invalid choice")

    def find_duplicates(self):
        """Main method to find all duplicates"""
        start_time = time.time()

        try:
            self.scan_directory()
            self.compare_filenames()
            self.compare_file_content()

            self.stats["time_taken"] = time.time() - start_time

            self.logger.info(
                f"Duplicate detection completed in {self.stats['time_taken']:.2f} seconds"
            )
            self.logger.info(
                f"Total duplicates found: {self.stats['duplicate_files']} files ({self._format_size(self.stats['duplicate_size'])})"
            )

            # Mark as completed in checkpoint
            self.current_stage = "completed"
            self._save_checkpoint()

            return {
                "filename_duplicates": self.filename_duplicates,
                "content_duplicates": self.content_duplicates,
                "stats": self.stats,
            }
        except KeyboardInterrupt:
            # Handle Ctrl+C by saving progress
            elapsed = time.time() - start_time
            self.logger.warning("Process interrupted by user. Saving progress...")
            self._save_checkpoint()
            self.logger.info(f"Progress saved to {self.checkpoint_file}")
            self.logger.info(
                f"Resume with --resume flag to continue where you left off"
            )
            raise
        except Exception as e:
            # Save progress on unexpected errors
            self.logger.error(f"Error during duplicate detection: {str(e)}")
            self._save_checkpoint()
            self.logger.info(f"Progress saved to {self.checkpoint_file}")
            raise

    def _save_checkpoint(self):
        """Save current progress to checkpoint file"""
        try:
            checkpoint_data = {
                "directory": self.directory,
                "current_stage": self.current_stage,
                "size_groups": dict(
                    self.size_groups
                ),  # Convert defaultdict to regular dict
                "filename_duplicates": self.filename_duplicates,
                "content_duplicates": self.content_duplicates,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat(),
            }

            with open(self.checkpoint_file, "wb") as f:
                pickle.dump(checkpoint_data, f)

            # Also save a human-readable summary
            summary_file = os.path.join(
                self.checkpoint_dir, f"dedup_summary_{Path(self.directory).name}.json"
            )
            summary = {
                "directory": self.directory,
                "stage": self.current_stage,
                "stats": self.stats,
                "timestamp": datetime.now().isoformat(),
                "checkpoint_file": self.checkpoint_file,
            }
            with open(summary_file, "w", encoding="utf-8") as f:
                json.dump(summary, f, indent=2)

            self.logger.debug(f"Checkpoint saved to {self.checkpoint_file}")
        except Exception as e:
            self.logger.error(f"Failed to save checkpoint: {str(e)}")

    def _load_checkpoint(self):
        """Load progress from checkpoint file"""
        if not os.path.exists(self.checkpoint_file):
            self.logger.warning(f"No checkpoint file found at {self.checkpoint_file}")
            return False

        try:
            with open(self.checkpoint_file, "rb") as f:
                checkpoint_data = pickle.load(f)

            # Verify this checkpoint is for the same directory
            if checkpoint_data["directory"] != self.directory:
                self.logger.warning(
                    f"Checkpoint is for a different directory: {checkpoint_data['directory']}"
                )
                return False

            # Restore state
            self.current_stage = checkpoint_data["current_stage"]
            # Convert dict back to defaultdict
            self.size_groups = defaultdict(list)
            for size, files in checkpoint_data["size_groups"].items():
                self.size_groups[size] = files
            self.filename_duplicates = checkpoint_data["filename_duplicates"]
            self.content_duplicates = checkpoint_data["content_duplicates"]
            self.stats = checkpoint_data["stats"]

            self.logger.info(f"Resumed from checkpoint: {self.checkpoint_file}")
            self.logger.info(f"Current stage: {self.current_stage}")
            return True
        except Exception as e:
            self.logger.error(f"Failed to load checkpoint: {str(e)}")
            return False

    def _format_size(self, size_bytes):
        """Format file size in human readable format"""
        if size_bytes < 1024:
            return f"{size_bytes} bytes"
        elif size_bytes < 1024 * 1024:
            return f"{size_bytes / 1024:.2f} KB"
        elif size_bytes < 1024 * 1024 * 1024:
            return f"{size_bytes / (1024 * 1024):.2f} MB"
        else:
            return f"{size_bytes / (1024 * 1024 * 1024):.2f} GB"


def main():
    parser = argparse.ArgumentParser(description="Find and manage duplicate files")

    # Add directory as both positional and optional argument for flexibility
    directory_group = parser.add_mutually_exclusive_group(required=True)
    directory_group.add_argument(
        "directory", nargs="?", help="Directory to scan for duplicates", default=None
    )
    directory_group.add_argument(
        "-dir", "--directory", dest="dir_opt", help="Directory to scan for duplicates"
    )

    parser.add_argument(
        "-r", "--recursive", action="store_true", help="Scan directories recursively"
    )
    parser.add_argument("-d", "--max-depth", type=int, help="Maximum recursion depth")
    parser.add_argument("-e", "--exclude", nargs="+", help="File extensions to exclude")
    parser.add_argument(
        "-i", "--include", nargs="+", help="Only include these file extensions"
    )
    parser.add_argument(
        "-t",
        "--threads",
        type=int,
        default=os.cpu_count(),
        help="Number of threads to use",
    )
    parser.add_argument(
        "-a",
        "--action",
        choices=["report", "delete", "interactive"],
        default="report",
        help="Action to take with duplicates",
    )
    parser.add_argument(
        "-k",
        "--keep",
        choices=["newest", "oldest", "first_found"],
        default="newest",
        help="Strategy for selecting which duplicate to keep",
    )
    parser.add_argument(
        "-s",
        "--symlink",
        action="store_true",
        help="Create symlinks to kept files instead of deleting",
    )
    parser.add_argument(
        "-c", "--chunk-size", type=int, default=8192, help="Chunk size for file reading"
    )
    parser.add_argument(
        "--resume", action="store_true", help="Resume from last checkpoint if available"
    )
    parser.add_argument(
        "--clear-checkpoints",
        action="store_true",
        help="Clear all checkpoints before starting",
    )

    args = parser.parse_args()

    # Use whichever directory argument was provided
    directory = args.directory if args.directory else args.dir_opt

    # Clear checkpoints if requested
    if args.clear_checkpoints:
        checkpoint_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "checkpoints"
        )
        if os.path.exists(checkpoint_dir):
            for f in os.listdir(checkpoint_dir):
                if f.startswith("dedup_checkpoint_") or f.startswith("dedup_summary_"):
                    os.remove(os.path.join(checkpoint_dir, f))
            print("All checkpoints cleared.")

    # Normalize and validate exclude/include extensions
    exclude_exts = [
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (args.exclude or [])
    ]
    include_exts = [
        ext.lower() if ext.startswith(".") else f".{ext.lower()}"
        for ext in (args.include or [])
    ]

    finder = DuplicateFileFinder(
        directory=directory,
        recursive=args.recursive,
        exclude_extensions=exclude_exts,
        include_extensions=include_exts,
        max_depth=args.max_depth,
        chunk_size=args.chunk_size,
        num_threads=args.threads,
        keep_strategy=args.keep,
        resume=args.resume,
    )

    try:
        # Find duplicates
        finder.find_duplicates()

        # Process duplicates according to selected action
        finder.process_duplicates(action=args.action, symlink=args.symlink)
    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Progress has been saved.")
        print(f"Run with --resume flag to continue the scan.")
        exit(1)


if __name__ == "__main__":
    main()
