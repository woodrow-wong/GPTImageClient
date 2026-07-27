"""
Workflow: extract common elements from multiple images → generate outline drawing.

Usage:
    1. Put your reference images in ./input/ folder
    2. Set CODEX_API_KEY and CODEX_BASE_URL env vars
    3. Run: uv run python example_outline.py

The script will:
    Step 1: Analyze all images to find common elements (via vision model)
    Step 2: Generate a black-ink line drawing from the extracted description
"""

import os
import sys
from gpt_image_client import GPTImageClient


def main():
    api_key = os.environ.get("CODEX_API_KEY", "")
    base_url = os.environ.get("CODEX_BASE_URL", "")

    if not api_key:
        print("ERROR: CODEX_API_KEY environment variable is not set.")
        sys.exit(1)
    if not base_url:
        print("ERROR: CODEX_BASE_URL environment variable is not set.")
        sys.exit(1)

    client = GPTImageClient(api_key=api_key, base_url=base_url)

    # ── Configure your image paths here ──────────────────────────────────
    image_paths = [
        "./input/scene_v1.png",
        "./input/scene_v2.png",
        "./input/scene_v3.png",
    ]

    chat_model = "gpt-5.6-sol-max" # vision model for analysis (adjust to your proxy's actual model name)
    image_model = "gpt-image-2"    # model for outline generation (image models have no reasoning level)

    # ── Step 1: Extract common elements via vision analysis ──────────────
    print("Analyzing common elements across images...")
    description = client.extract_outline_description(
        image_paths=image_paths,
        chat_model=chat_model,
    )
    print(f"  Extracted description: {description}\n")

    # ── Step 2: Generate line-art outline drawing ────────────────────────
    print("Generating outline drawing (白描)...")
    images = client.generate_outline(
        description=description,
        model=image_model,
        size="1024x1024",
        n=2,
    )
    paths = client.save_all(images, output_dir="./output", prefix="outline")
    for p in paths:
        print(f"  Saved: {p}")
    print("Done. Send the best outline to your craft artist!")


if __name__ == "__main__":
    main()
