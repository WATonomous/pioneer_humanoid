#!/usr/bin/env python3
"""
Test script to verify Hugging Face repo creation and file upload.

Usage:
    export HF_TOKEN="hf_your_write_token"
    python3 test_hf_upload.py

Or pass token directly:
    python3 test_hf_upload.py --token "hf_your_write_token"
"""

import os
import sys
import argparse
import tempfile
from pathlib import Path

try:
    from huggingface_hub import HfApi
except ImportError:
    print("Error: huggingface_hub is not installed. Run: pip install huggingface_hub")
    sys.exit(1)


def load_env():
    """Auto-load variables from .env if present."""
    env_paths = [
        Path(__file__).parent / ".env",
        Path.cwd() / ".env",
        Path.cwd() / "ultron" / ".env",
    ]
    for env_path in env_paths:
        if env_path.exists():
            for line in env_path.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#"):
                    continue
                if "=" in line:
                    k, v = line.split("=", 1)
                    k = k.strip()
                    v = v.strip().strip("'\"")
                    if k and k not in os.environ:
                        os.environ[k] = v


def main():
    load_env()
    parser = argparse.ArgumentParser(description="Test Hugging Face repo creation & upload")
    parser.add_argument("--token", default=None, help="Hugging Face User Access Token (write permission)")
    parser.add_argument("--repo-name", default="ultron-hf-test", help="Name of repository to create")
    parser.add_argument("--private", action="store_true", default=True, help="Make repository private (default: True)")
    args = parser.parse_args()

    # 1. Resolve token
    token = args.token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_HUB_TOKEN")
    if not token:
        print("\n" + "=" * 60)
        print("❌ No Hugging Face token found in environment or .env!")
        print("=" * 60)
        print("Make sure HF_TOKEN=hf_... is set inside your ultron/.env file.")
        print("=" * 60 + "\n")
        sys.exit(1)

    api = HfApi(token=token)

    # 2. Verify credentials
    print("Connecting to Hugging Face...")
    try:
        user_info = api.whoami()
        user_name = user_info["name"]
        print(f"✓ Authenticated as: {user_name}")
    except Exception as e:
        print(f"❌ Authentication failed: {e}")
        sys.exit(1)

    repo_id = f"{user_name}/{args.repo_name}"
    print(f"\nTarget repository: https://huggingface.co/{repo_id}")

    # 3. Create repository
    print(f"Creating repository '{repo_id}' (private={args.private})...")
    try:
        repo_url = api.create_repo(
            repo_id=repo_id,
            repo_type="model",
            private=args.private,
            exist_ok=True,
        )
        print(f"✓ Repository ready: {repo_url}")
    except Exception as e:
        print(f"❌ Failed to create repo: {e}")
        sys.exit(1)

    # 4. Create temporary test files
    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)

        # random.py with one line of code
        random_py = tmp_path / "random.py"
        random_py.write_text('print("Hello from Ultron AI pipeline!")\n')

        # Dummy model file (test_model.pt)
        dummy_model = tmp_path / "test_model.pt"
        try:
            import torch
            torch.save({"dummy_weights": torch.randn(10, 10), "epoch": 1, "loss": 0.042}, dummy_model)
            print("✓ Created dummy PyTorch weights file (test_model.pt)")
        except ImportError:
            # Fallback if torch isn't installed in this specific env
            dummy_model.write_bytes(b"DUMMY_MODEL_WEIGHTS_BINARY_BLOB_FOR_TESTING\n" * 50)
            print("✓ Created dummy binary model file (test_model.pt)")

        # README.md for the model card
        readme = tmp_path / "README.md"
        readme.write_text(f"""---
tags:
- robotics
- imitation-learning
- ultron-test
---

# {args.repo_name}

Test repository created by Ultron AI Agent to verify Hugging Face integration.

- Contains `random.py`
- Contains `test_model.pt`
""")

        # 5. Upload files to Hugging Face
        print(f"\nUploading files from {tmpdir} to {repo_id}...")
        try:
            api.upload_folder(
                folder_path=str(tmp_path),
                repo_id=repo_id,
                repo_type="model",
                commit_message="Test upload: random.py and test_model.pt",
            )
            print("\n" + "=" * 60)
            print("🎉 SUCCESS! Files uploaded successfully!")
            print(f"View your repository at: https://huggingface.co/{repo_id}")
            print("=" * 60 + "\n")
        except Exception as e:
            print(f"❌ Upload failed: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()
