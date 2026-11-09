import argparse
import os
import shutil
import subprocess
import zipfile
from pathlib import Path

import requests
from datasets import load_dataset
from tqdm import tqdm


# Project root: rag-security-research/
PROJECT_ROOT = Path(__file__).resolve().parents[3]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
BEIR_DIR = RAW_DIR / "beir_nq"
NQ_OPEN_DIR = RAW_DIR / "nq_open"


def download_file_resumable(url: str, dest_path: Path, chunk_size: int = 2 * 1024 * 1024) -> Path:
    """Download a file with HTTP Range support for automatic resume and large chunk buffers."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_name(dest_path.name + ".part")

    existing_bytes = temp_path.stat().st_size if temp_path.exists() else 0

    headers = {"User-Agent": "Mozilla/5.0"}
    if existing_bytes > 0:
        headers["Range"] = f"bytes={existing_bytes}-"
        print(f"Resuming download from byte {existing_bytes:,} ({existing_bytes / (1024 * 1024):.1f} MB)...")

    response = requests.get(url, headers=headers, stream=True, timeout=60)

    # 416 means Requested Range Not Satisfiable (file is already fully downloaded in temp_path)
    if response.status_code == 416:
        temp_path.rename(dest_path)
        return dest_path

    if response.status_code not in (200, 206):
        raise RuntimeError(f"Failed to download {url}: HTTP {response.status_code}")

    content_length = response.headers.get("Content-Length")
    total_bytes = (int(content_length) + existing_bytes) if content_length else None

    mode = "ab" if existing_bytes > 0 and response.status_code == 206 else "wb"
    if mode == "wb":
        existing_bytes = 0

    with open(temp_path, mode) as f, tqdm(
        desc=dest_path.name,
        total=total_bytes,
        initial=existing_bytes,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as pbar:
        for chunk in response.iter_content(chunk_size=chunk_size):
            if chunk:
                f.write(chunk)
                pbar.update(len(chunk))

    temp_path.rename(dest_path)
    return dest_path


def download_with_curl(url: str, dest_path: Path) -> bool:
    """Attempt download using Windows built-in curl with auto-resume."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    part_path = dest_path.with_name(dest_path.name + ".part")
    curl_bin = shutil.which("curl") or shutil.which("curl.exe")
    if not curl_bin:
        return False

    cmd = [
        curl_bin,
        "-C", "-",           # Auto-resume from existing bytes
        "-L",                # Follow redirects
        "--retry", "5",      # Retry up to 5 times on drop
        "--retry-delay", "2",
        "--connect-timeout", "30",
        "-o", str(part_path),
        url,
    ]
    try:
        print(f"Downloading with system curl (resumable) to {part_path.name}...")
        res = subprocess.run(cmd)
        if res.returncode == 0 and part_path.exists():
            if dest_path.exists():
                dest_path.unlink()
            part_path.rename(dest_path)
            return True
    except Exception as e:
        print(f"curl invocation failed ({e}), falling back to Python downloader...")
    return False


def unzip_beir_archive(zip_path: Path, extract_to: Path) -> Path:
    """Extract BEIR zip archive and organize files directly under extract_to."""
    extract_to.mkdir(parents=True, exist_ok=True)
    print(f"\nExtracting {zip_path.name} to {extract_to}...")

    temp_extract = extract_to.parent / "_temp_extract_beir"
    if temp_extract.exists():
        shutil.rmtree(temp_extract)
    temp_extract.mkdir(parents=True, exist_ok=True)

    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(temp_extract)

    # BEIR archives contain a top-level directory named after the dataset (e.g. 'nq')
    extracted_items = list(temp_extract.iterdir())
    source_dir = extracted_items[0] if len(extracted_items) == 1 and extracted_items[0].is_dir() else temp_extract

    for item in source_dir.iterdir():
        target = extract_to / item.name
        if target.exists():
            if target.is_dir():
                shutil.rmtree(target)
            else:
                target.unlink()
        shutil.move(str(item), str(target))

    shutil.rmtree(temp_extract, ignore_errors=True)
    print(f"Extraction complete.")
    return extract_to


def download_beir_nq(use_curl: bool = True):
    """Download and extract the original BEIR Natural Questions dataset with resume support."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    BEIR_DIR.mkdir(parents=True, exist_ok=True)

    # Check if already present
    if (BEIR_DIR / "corpus.jsonl.gz").exists():
        print(f"\nBEIR NQ already present at: {BEIR_DIR}")
        return BEIR_DIR

    dataset_url = "https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/nq.zip"
    zip_path = RAW_DIR / "nq.zip"

    if not zip_path.exists() or zip_path.stat().st_size < 100_000:
        print(f"\nDownloading BEIR NQ archive (498 MB)...")
        downloaded = False
        if use_curl:
            downloaded = download_with_curl(dataset_url, zip_path)
        if not downloaded:
            download_file_resumable(dataset_url, zip_path, chunk_size=2 * 1024 * 1024)

    unzip_beir_archive(zip_path, BEIR_DIR)

    print("\nBEIR files ready:")
    for file_path in sorted(BEIR_DIR.rglob("*")):
        if file_path.is_file():
            print(f"  {file_path.relative_to(BEIR_DIR)} ({file_path.stat().st_size / (1024*1024):.2f} MB)")

    return BEIR_DIR


def download_nq_open():
    """Download the original NQ-Open validation split via Hugging Face datasets."""
    NQ_OPEN_DIR.mkdir(parents=True, exist_ok=True)
    output_file = NQ_OPEN_DIR / "validation.jsonl"

    if output_file.exists() and output_file.stat().st_size > 0:
        print(f"\nNQ Open already present at: {output_file}")
        return output_file

    print("\nDownloading NQ Open validation split...")
    dataset = load_dataset(
        "google-research-datasets/nq_open",
        split="validation",
    )

    dataset.to_json(
        str(output_file),
        orient="records",
        lines=True,
    )

    print(f"NQ Open saved to: {output_file}")
    print(f"Number of examples: {len(dataset)}")
    print(f"Available fields: {dataset.column_names}")

    return output_file


def main():
    parser = argparse.ArgumentParser(description="Download and prepare datasets for RAG Security Baseline.")
    parser.add_argument(
        "--no-curl",
        action="store_true",
        help="Disable curl and use Python requests with 2MB chunk buffer and resume support.",
    )
    args = parser.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Step 1: Downloading NQ Open Validation Split")
    print("=" * 60)
    download_nq_open()

    print("\n" + "=" * 60)
    print("Step 2: Downloading BEIR Natural Questions (NQ)")
    print("=" * 60)
    download_beir_nq(use_curl=not args.no_curl)

    print("\nDataset downloads and checks completed.")


if __name__ == "__main__":
    main()
