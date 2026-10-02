"""
comma2k19 Dataset Download Script.

Downloads the comma2k19 dataset (~95 GB in 10 chunks) from comma.ai's
Hugging Face dataset. Run this before Phase 2 training, then decode it once
with scripts/preprocess_comma2k19.py.

Usage:
    python scripts/download_comma2k19.py --output-dir data/comma2k19
    python scripts/download_comma2k19.py --output-dir data/comma2k19 --chunks 1    # ~9 GB
"""

import argparse
from pathlib import Path


# comma.ai publishes the raw chunks on Hugging Face as raw_data/Chunk_<n>.zip
# (8.7-9.9 GB each). The repo also holds other material, so a full snapshot
# download would fetch far more than the chunks.
HUGGINGFACE_REPO = "commaai/comma2k19"
NUM_CHUNKS = 10


def download_chunks(output_dir: str, chunks: list[int] = None, keep_zip: bool = False):
    """
    Download and extract chunks to ``<output_dir>/Chunk_<n>/<route>/<segment>``.

    Chunks already extracted are skipped; interrupted downloads resume.
    """
    import zipfile

    from huggingface_hub import hf_hub_download

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    chunks = chunks or list(range(1, NUM_CHUNKS + 1))

    for n in chunks:
        chunk_dir = output_path / f"Chunk_{n}"
        if chunk_dir.is_dir() and any(chunk_dir.iterdir()):
            print(f"Chunk {n} already extracted, skipping.")
            continue
        print(f"Downloading Chunk {n} (~9 GB) from huggingface.co/datasets/{HUGGINGFACE_REPO}...")
        archive = Path(hf_hub_download(
            repo_id=HUGGINGFACE_REPO,
            repo_type="dataset",
            filename=f"raw_data/Chunk_{n}.zip",
            local_dir=output_path / "_downloads",
        ))
        with zipfile.ZipFile(archive) as zf:
            # Archives may or may not contain the Chunk_<n>/ folder itself.
            has_root = all(name.startswith(f"Chunk_{n}/") for name in zf.namelist())
            print(f"Extracting {archive.name}...")
            zf.extractall(output_path if has_root else chunk_dir)
        if not keep_zip:
            archive.unlink()

    print("\nDone. Verify with: python scripts/download_comma2k19.py --verify")


def verify_dataset(dataset_dir: str):
    """
    Verify downloaded dataset structure and CAN field completeness.

    Checks:
    - Segment directories exist
    - video.hevc present
    - processed_log/CAN/steering_angle/{t,value} present
    - processed_log/CAN/speed/{t,value} present (m/s)
    """
    dataset_path = Path(dataset_dir)

    if not dataset_path.exists():
        print(f"Dataset directory not found: {dataset_dir}")
        return False

    total_segments = 0
    total_videos = 0
    total_complete = 0  # segments with all required files
    missing_steering = 0
    missing_speed = 0
    missing_video = 0

    required_can_fields = {
        "steering_angle": ["t", "value"],
        "speed": ["t", "value"],
    }

    for chunk_dir in sorted(dataset_path.iterdir()):
        if not chunk_dir.is_dir() or chunk_dir.name.startswith("_"):
            continue

        chunk_segments = 0
        chunk_complete = 0

        for route_dir in chunk_dir.iterdir():
            if not route_dir.is_dir():
                continue
            for segment_dir in route_dir.iterdir():
                if not segment_dir.is_dir():
                    continue
                chunk_segments += 1
                total_segments += 1

                has_video = (segment_dir / "video.hevc").exists()
                if has_video:
                    total_videos += 1
                else:
                    missing_video += 1

                # Check CAN fields
                processed_log = segment_dir / "processed_log"
                has_steering = all(
                    (processed_log / "CAN" / "steering_angle" / f).exists()
                    for f in required_can_fields["steering_angle"]
                )
                has_speed = all(
                    (processed_log / "CAN" / "speed" / f).exists()
                    for f in required_can_fields["speed"]
                )

                if not has_steering:
                    missing_steering += 1
                if not has_speed:
                    missing_speed += 1

                if has_video and has_steering and has_speed:
                    chunk_complete += 1
                    total_complete += 1

        print(f"  {chunk_dir.name}: {chunk_segments} segments, {chunk_complete} complete")

    print(f"\n{'='*40}")
    print(f"Total segments:     {total_segments}")
    print(f"Complete segments:  {total_complete}")
    print(f"Missing video:      {missing_video}")
    print(f"Missing steering:   {missing_steering}")
    print(f"Missing speed:      {missing_speed}")
    print(f"Expected:           ~2019 segments")

    if total_complete >= 1000:
        print("\n✓ Dataset looks complete enough for training.")
        return True
    elif total_complete > 0:
        print("\n⚠️  Partial dataset. Enough for development, may need more for full training.")
        return True
    else:
        print("\n✗ No complete segments found. Check download.")
        return False


def main():
    parser = argparse.ArgumentParser(description="Download comma2k19 Dataset")
    parser.add_argument("--output-dir", default="data/comma2k19",
                        help="Directory to download dataset to")
    parser.add_argument("--chunks", nargs="+", type=int, default=None,
                        help="Specific chunk numbers to download (1-10)")
    parser.add_argument("--keep-zip", action="store_true",
                        help="Keep the downloaded archives after extraction")
    parser.add_argument("--verify", action="store_true",
                        help="Verify existing download")
    args = parser.parse_args()

    if args.verify:
        verify_dataset(args.output_dir)
        return

    download_chunks(args.output_dir, args.chunks, keep_zip=args.keep_zip)


if __name__ == "__main__":
    main()
