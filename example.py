"""
Usage example for GPTImageClient.

Set environment variables:
    CODEX_API_KEY  - Your Codex proxy API key
    CODEX_BASE_URL - Your Codex proxy base URL

Or pass them directly to GPTImageClient(api_key=..., base_url=...).
"""

import os
import sys
from gpt_image_client import GPTImageClient


def main():
    api_key = os.environ.get("CODEX_API_KEY", "")
    base_url = os.environ.get("CODEX_BASE_URL", "")

    if not api_key:
        print("ERROR: CODEX_API_KEY environment variable is not set.")
        print('  $env:CODEX_API_KEY = "your-api-key"')
        print("  Or edit this script and pass api_key=... directly.")
        sys.exit(1)

    if not base_url:
        print("ERROR: CODEX_BASE_URL environment variable is not set.")
        print('  $env:CODEX_BASE_URL = "https://your-codex-proxy.com"')
        print("  Or edit this script and pass base_url=... directly.")
        sys.exit(1)

    client = GPTImageClient(api_key=api_key, base_url=base_url)

    # ── 1. Text-to-image generation ─────────────────────────────────────
    print("Generating images...")
    images = client.generate(
        prompt="A serene mountain lake at sunset, digital painting style",
        model="gpt-image-2",
        n=1,
        size="1024x1024",
        quality="standard",
        style="vivid",
        response_format="b64_json",
    )
    paths = client.save_all(images, output_dir="./output", prefix="gen")
    for p in paths:
        print(f"  Saved: {p}")

    # ── 2. Image variation ──────────────────────────────────────────────
    # source_img = "./input/source.png"   # square PNG, <4MB
    # print("Creating variations...")
    # variations = client.variation(
    #     image_path=source_img,
    #     n=2,
    #     size="1024x1024",
    # )
    # client.save_all(variations, output_dir="./output", prefix="var")

    # ── 3. Image editing (inpainting) ───────────────────────────────────
    # source_img = "./input/source.png"
    # mask_img = "./input/mask.png"       # transparent = editable area
    # print("Editing image...")
    # edits = client.edit(
    #     image_path=source_img,
    #     prompt="Add a rainbow in the sky",
    #     mask_path=mask_img,
    #     n=1,
    # )
    # client.save_all(edits, output_dir="./output", prefix="edit")


if __name__ == "__main__":
    main()
