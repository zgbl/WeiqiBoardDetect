import pandas as pd
import os
import shutil
from pathlib import Path
import argparse

def main():
    parser = argparse.ArgumentParser(description="Clean dirty data by moving suspicious samples to a backup folder.")
    parser.add_argument("--csv",     default="dirty_data_report/dirty_data_candidates.csv", help="Path to dirty data CSV")
    parser.add_argument("--backup",  default="dirty_patches_backup",                      help="Directory to move dirty images to")
    parser.add_argument("--dry_run", action="store_true",                                  help="Show what would be moved without moving")
    args = parser.parse_args()

    if not os.path.exists(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        return

    df = pd.read_csv(args.csv)
    # Filter for suspicious samples
    suspicious = df[df["is_suspicious"] == True]
    
    if len(suspicious) == 0:
        print("[DONE] No suspicious samples found in the report.")
        return

    print(f"[INFO] Found {len(suspicious):,} suspicious samples to move.")
    backup_path = Path(args.backup)
    
    count = 0
    for _, row in suspicious.iterrows():
        src = Path(row["path"])
        if not src.exists():
            continue
            
        # Task: Move to a 'DIRTY' subdirectory within the SAME CLASS folder
        # e.g., patches/B/patch1.jpg -> patches/B/DIRTY/patch1.jpg
        dest_dir = src.parent / "DIRTY"
        dest_dir.mkdir(parents=True, exist_ok=True)
        
        dest = dest_dir / src.name
        
        if args.dry_run:
            if count < 10:
                print(f"[DRY-RUN] Move: {src} -> {dest}")
        else:
            try:
                shutil.move(str(src), str(dest))
            except Exception as e:
                print(f"[ERROR] Failed to move {src}: {e}")
        
        count += 1

    if args.dry_run:
        print(f"\n[DONE] Dry-run complete. Would have moved {count:,} files.")
    else:
        print(f"\n[DONE] Successfully moved {count:,} files into their respective DIRTY subfolders.")

if __name__ == "__main__":
    main()
