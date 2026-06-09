#!/usr/bin/env python
"""Script to delete all experiment result directories from models."""

import shutil
from pathlib import Path


def delete_directory(path: Path, description: str):
    """Delete a directory and all its contents."""
    if not path.exists():
        print(f"  {description}: {path} does not exist, skipping...")
        return
    
    if not path.is_dir():
        print(f"  {description}: {path} is not a directory, skipping...")
        return
    
    try:
        shutil.rmtree(path)
        print(f"  ✓ Deleted {description}: {path}")
    except Exception as e:
        print(f"  ✗ Error deleting {description} {path}: {e}")


def delete_experiment_directories(task_path: Path, task_name: str):
    """Delete all experiment result directories for a specific task."""
    if not task_path.exists():
        print(f"\n{task_name}: directory does not exist, skipping...")
        return
    
    print(f"\nDeleting {task_name} experiment results...")
    
    # Find all experiment directories (those matching the pattern *_nsteps=*_ninter=*_loss=*)
    deleted_count = 0
    for item in task_path.iterdir():
        if item.is_dir():
            delete_directory(item, f"{task_name} experiment: {item.name}")
            deleted_count += 1
    
    if deleted_count == 0:
        print(f"  No experiment directories found for {task_name}")


def main():
    """Main function to delete all experiment result directories."""
    base_dir = Path("models")
    
    print("=" * 60)
    print("Deleting experiment result directories from models")
    print("=" * 60)
    
    # Delete fhn, vdp1, vdp2 experiment results
    tasks = ["fhn", "vdp1", "vdp2"]
    for task in tasks:
        task_path = base_dir / task
        delete_experiment_directories(task_path, task)
    
    # Delete mocap - has multiple subdirectories
    print("\n" + "=" * 60)
    print("Deleting mocap experiment result directories...")
    print("=" * 60)
    
    # Find all mocap subdirectories (mocap09short, mocap09long, etc.)
    for item in base_dir.iterdir():
        if item.is_dir() and item.name.startswith("mocap"):
            delete_experiment_directories(item, item.name)
    
    print("\n" + "=" * 60)
    print("Done!")
    print("=" * 60)


if __name__ == "__main__":
    quit()   # DONT WANT TO ACCIDENTALLY DELETE RESULTS AGAIN
    main()
