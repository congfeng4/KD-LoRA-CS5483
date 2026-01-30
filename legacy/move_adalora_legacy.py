#!/usr/bin/env python3
"""
Move existing AdaLoRA results to legacy directory for re‑running experiments.
"""

import os
import shutil
import argparse
from pathlib import Path

def move_adalora_results(dry_run=False):
    base_dir = Path("./results")
    legacy_dir = base_dir / "legacy" / "adalora"
    if not dry_run:
        legacy_dir.mkdir(parents=True, exist_ok=True)
    
    # Patterns to match: any directory containing 'adalora' in its name
    patterns = ["kd-lora", "lora"]  # subdirectories under results
    
    moved_count = 0
    for pattern in patterns:
        source_dir = base_dir / pattern
        if not source_dir.exists():
            continue
            
        # Walk through all subdirectories
        for root, dirs, files in os.walk(str(source_dir)):
            root_path = Path(root)
            for dir_name in dirs:
                if "adalora" in dir_name.lower():
                     src = root_path / dir_name
                     # Preserve relative path under legacy/adalora/
                     rel_path = src.relative_to(base_dir)
                     dst = legacy_dir / rel_path
                     
                     # Ensure destination parent exists
                     if not dry_run:
                         dst.parent.mkdir(parents=True, exist_ok=True)
                     
                     if dry_run:
                         print(f"[DRY RUN] Would move: {src} -> {dst}")
                         moved_count += 1
                     else:
                         print(f"Moving: {src} -> {dst}")
                         try:
                             shutil.move(str(src), str(dst))
                             moved_count += 1
                         except Exception as e:
                             print(f"  Error moving {src}: {e}")
    
    if dry_run:
        print(f"\nWould move {moved_count} AdaLoRA directories to {legacy_dir}")
    else:
        print(f"\nMoved {moved_count} AdaLoRA directories to {legacy_dir}")
    
    # Also move any stray JSON files with 'adalora' in the name
    json_count = 0
    for json_file in base_dir.rglob("*.json"):
        if "adalora" in json_file.name.lower():
            rel_path = json_file.relative_to(base_dir)
            dst = legacy_dir / rel_path
            if not dry_run:
                dst.parent.mkdir(parents=True, exist_ok=True)
            if dry_run:
                print(f"[DRY RUN] Would move JSON: {json_file} -> {dst}")
                json_count += 1
            else:
                print(f"Moving JSON: {json_file} -> {dst}")
                try:
                    shutil.move(str(json_file), str(dst))
                    json_count += 1
                except Exception as e:
                    print(f"  Error moving {json_file}: {e}")
    
    if dry_run:
        print(f"Would move {json_count} AdaLoRA JSON files")
    else:
        print(f"Moved {json_count} AdaLoRA JSON files")
    
    # Summary
    total = moved_count + json_count
    if total > 0:
        if dry_run:
            print(f"\nTotal would move: {total} items")
        else:
            print(f"\nTotal moved: {total} items")
        print(f"Legacy directory: {legacy_dir}")
    else:
        if dry_run:
            print("\nNo AdaLoRA results found to move (dry-run).")
        else:
            print("\nNo AdaLoRA results found to move.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Move existing AdaLoRA results to legacy directory")
    parser.add_argument("--dry-run", action="store_true", help="Print what would be moved without actually moving")
    args = parser.parse_args()
    move_adalora_results(dry_run=args.dry_run)