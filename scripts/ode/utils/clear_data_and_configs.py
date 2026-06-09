#!/usr/bin/env python
"""Script to clear all data and config files from experiments directories."""

import shutil
from pathlib import Path


def clear_directory(path: Path, description: str, verbose: bool=False):
    """Remove all contents of a directory but keep the directory itself."""
    if not path.exists():
        print(f"  {description}: {path} does not exist, skipping...")
        return
    
    if not path.is_dir():
        print(f"  {description}: {path} is not a directory, skipping...")
        return
    
    try:
        # Remove all contents
        for item in path.iterdir():
            if item.is_file():
                item.unlink()
                if verbose:
                    print(f"    Deleted file: {item}")
            elif item.is_dir():
                #shutil.rmtree(item)
                if verbose:
                    print(f"    Kept directory: {item}")
                clear_directory(item, str(item))
        if verbose:
            print(f"  ✓ Cleared {description}: {path}")
    except Exception as e:
        if verbose:
            print(f"  ✗ Error clearing {description} {path}: {e}")


def clear_data_and_configs_for_task(base_path: Path, task_name: str, verbose:bool=False):
    """Clear configs and data directories for a specific task."""
    print(f"\nClearing {task_name}...")
    
    configs_path = base_path / "configs"
    data_path = base_path / "data"
    
    clear_directory(configs_path, f"{task_name} configs", verbose=verbose)
    clear_directory(data_path, f"{task_name} data", verbose=verbose)


def clear_data_and_configs(verbose=False):
    """Main function to clear all data and configs."""
    base_dir = Path("experiments")
    
    print("=" * 60)
    print("Clearing data and config files from experiments")
    print("=" * 60)
    
    # Clear fhn, vdp1, vdp2
    tasks = ["fhn", "vdp1", "vdp2"]
    for task in tasks:
        task_path = base_dir / task
        if task_path.exists():
            clear_data_and_configs_for_task(task_path, task, verbose=verbose)
        elif verbose:
            print(f"\n{task}: directory does not exist, skipping...")
    
    # Clear mocap - has multiple subdirectories
    if verbose:
        print("\n" + "=" * 60)
        print("Clearing mocap data and configs...")
        print("=" * 60)
    
    mocap_dir = base_dir / "mocap"
    if not mocap_dir.exists() and verbose:
        print("mocap directory does not exist, skipping...")
    else:
        # Find all mocap subdirectories (mocap09short, mocap09long, etc.)
        for item in mocap_dir.iterdir():
            if item.is_dir() and item.name.startswith("mocap"):
                # Clear configs and data for each mocap variant
                configs_path = item / "configs"
                data_path = item / "data"
                
                clear_directory(configs_path, f"{item.name} configs")
                clear_directory(data_path, f"{item.name} data")
                
                # Also check for any config.yaml files directly in the subdirectory
                config_file = item / "config.yaml"
                if config_file.exists():
                    config_file.unlink()
                    if verbose:
                        print(f"    Deleted file: {config_file}")
    
    if verbose:
        print("\n" + "=" * 60)
        print("Done!")
        print("=" * 60)


if __name__ == "__main__":
    clear_data_and_configs()
