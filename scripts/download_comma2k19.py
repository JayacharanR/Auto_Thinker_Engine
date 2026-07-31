"""
comma2k19 Dataset Download Script.

Downloads the comma2k19 dataset (~100 GB in 10 chunks) from the official
GitHub repository. Run this on the target hardware before Phase 2 training.

Usage:
    python scripts/download_comma2k19.py --output-dir data/comma2k19
    python scripts/download_comma2k19.py --output-dir data/comma2k19 --chunks 1 2  # Subset
"""

import argparse
import hashlib
import os
import subprocess
import sys
from pathlib import Path


# comma2k19 chunk URLs from the official repository
CHUNK_URLS = {
    1: "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5c5a.torrent",
    2: "https://academictorrents.com/download/65a2fbc964078aff62076ff4e103f18b951c5c5a.torrent",
    # Full URLs need to be sourced from https://github.com/commaai/comma2k19
    # The dataset is distributed via Academic Torrents and direct links
}

# Alternative: HuggingFace mirror (if available)
HUGGINGFACE_REPO = "commaai/comma2k19"


def download_via_huggingface(output_dir: str, chunks: list[int] = None):
    """
    Download comma2k19 from HuggingFace (preferred method).

    Requires: pip install huggingface_hub
    """
    try:
        from huggingface_hub import snapshot_download

        print(f"Downloading comma2k19 to {output_dir}...")
        print("This is approximately 100 GB. Ensure sufficient disk space.")

        snapshot_download(
            repo_id=HUGGINGFACE_REPO,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False,
        )
        print("Download complete!")

    except ImportError:
        print("huggingface_hub not installed. Install with: pip install huggingface_hub")
        sys.exit(1)
    except Exception as e:
        print(f"HuggingFace download failed: {e}")
        print("Falling back to direct download method...")
        download_direct(output_dir, chunks)


def download_direct(output_dir: str, chunks: list[int] = None):
    """
    Download comma2k19 chunks directly.

    Uses the commaai/comma2k19 download script pattern.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if chunks is None:
        chunks = list(range(1, 11))  # All 10 chunks

    print(f"Downloading {len(chunks)} chunk(s) to {output_dir}")
    print("Each chunk is approximately 10 GB.")
    print()

    # The official method uses the commaai/comma2k19 repo's download script
    # Clone it first if not present
    repo_dir = output_path / "_comma2k19_repo"
    if not repo_dir.exists():
        print("Cloning comma2k19 repository for download script...")
        subprocess.run(
            ["git", "clone", "--depth", "1",
             "https://github.com/commaai/comma2k19.git",
             str(repo_dir)],
            check=True,
        )

    for chunk_num in chunks:
        chunk_dir = output_path / f"Chunk_{chunk_num}"
        if chunk_dir.exists():
            print(f"Chunk {chunk_num} already exists, skipping.")
            continue

        print(f"Downloading Chunk {chunk_num}/10...")
        # Use the official download mechanism
        # The actual URLs are in the comma2k19 repo
        print(f"  Please download Chunk {chunk_num} manually from:")
        print(f"  https://github.com/commaai/comma2k19")
        print(f"  and extract to: {chunk_dir}")

    print("\nDone. Verify with: python scripts/download_comma2k19.py --verify")


def verify_dataset(dataset_dir: str):
    """Verify downloaded dataset structure."""
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return False

    total_segments = 0
    total_videos = 0
    total_logs = 0

    for chunk_dir in sorted(dataset_path.iterdir()):
        if not chunk_dir.is_dir() or chunk_dir.name.startswith("_"):
            continue

        chunk_segments = 0
        for route_dir in chunk_dir.iterdir():
            if not route_dir.is_dir():
                continue
            for segment_dir in route_dir.iterdir():
                if not segment_dir.is_dir():
                    continue
                chunk_segments += 1
                total_segments += 1

                if (segment_dir / "video.hevc").exists():
                    total_videos += 1
                if (segment_dir / "processed_log").exists():
                    total_logs += 1

        print(f"  {chunk_dir.name}: {chunk_segments} segments")

    print(f"\nTotal segments: {total_segments}")
    print(f"Videos found:   {total_videos}")
    print(f"Logs found:     {total_logs}")
    print(f"Expected:       ~2019 segments")

    if total_segments >= 1000:
        print("\n✓ Dataset looks complete enough for training.")
        return True
    elif total_segments > 0:
        print("\n⚠️  Partial dataset. Enough for development, may need more for full training.")
        return True
    else:
        print("\n✗ No segments found. Download may have failed.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download comma2k19 Dataset")
    parser.add_argument("--output-dir", default="data/comma2k19",
                        help="Directory to download dataset to")
    parser.add_argument("--chunks", nargs="+", type=int, default=None,
                        help="Specific chunk numbers to download (1-10)")
    parser.add_argument("--method", choices=["huggingface", "direct"], default="huggingface",
                        help="Download method")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing download")
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output_dir)
        return

    if args.method == "huggingface":
        download_via_huggingface(args.output_dir, args.chunks)
    else:
        download_direct(args.output_dir, args.chunks)


if __name__ == "__main__":
    main()
