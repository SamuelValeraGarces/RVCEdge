"""
Download base models for RVCEdge (one-time setup).

Downloads:
  - base_models/rmvpe.pt       (~100MB) — RMVPE pitch extractor
  - ContentVec auto-downloads on first use via transformers cache
"""

from pathlib import Path
from huggingface_hub import hf_hub_download

BASE_DIR = Path("base_models")
BASE_DIR.mkdir(exist_ok=True)

RMVPE_REPO = "lj1995/VoiceConversionWebUI"
RMVPE_FILE = "rmvpe.pt"


def download_rmvpe():
    dest = BASE_DIR / RMVPE_FILE
    if dest.exists():
        print(f"rmvpe.pt already exists ({dest.stat().st_size / 1e6:.1f}MB)")
        return
    print("Downloading rmvpe.pt from HuggingFace...")
    hf_hub_download(
        repo_id=RMVPE_REPO,
        filename=RMVPE_FILE,
        local_dir=str(BASE_DIR),
        local_dir_use_symlinks=False,
    )
    print(f"rmvpe.pt saved to {dest}")


def pre_cache_contentvec():
    """Pre-download ContentVec so first inference isn't slow."""
    print("Pre-caching ContentVec model (lengyue233/content-vec-best)...")
    print("This downloads ~360MB on first run.")
    try:
        from transformers import HubertModel, Wav2Vec2FeatureExtractor
        Wav2Vec2FeatureExtractor.from_pretrained("lengyue233/content-vec-best")
        HubertModel.from_pretrained("lengyue233/content-vec-best")
        print("ContentVec cached.")
    except Exception as e:
        print(f"ContentVec pre-cache failed: {e}")
        print("It will download automatically on first inference.")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-contentvec", action="store_true",
                        help="Skip pre-caching ContentVec (will download on first use)")
    args = parser.parse_args()

    download_rmvpe()
    if not args.skip_contentvec:
        pre_cache_contentvec()

    print("\nBase models ready.")
    print(f"Files in {BASE_DIR.resolve()}:")
    for f in BASE_DIR.rglob("*"):
        if f.is_file():
            print(f"  {f.name}: {f.stat().st_size / 1e6:.1f}MB")
