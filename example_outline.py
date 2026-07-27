"""Generate an outline drawing from a prepared design description.

Prepare ``outline_description.txt`` separately by analyzing the five reference
images with a vision-capable model, then run this script.
"""

import os
import sys
from pathlib import Path

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

    description_path = Path("outline_description.txt")
    if not description_path.is_file():
        print(f"ERROR: {description_path} was not found.")
        print("Create this file with your externally prepared design description.")
        sys.exit(1)

    description = description_path.read_text(encoding="utf-8").strip()
    if not description:
        print(f"ERROR: {description_path} is empty.")
        sys.exit(1)

    try:
        image_timeout = int(os.environ.get("IMAGE_TIMEOUT_SECONDS", "600"))
    except ValueError:
        print("ERROR: IMAGE_TIMEOUT_SECONDS must be an integer.")
        sys.exit(1)
    if image_timeout < 1:
        print("ERROR: IMAGE_TIMEOUT_SECONDS must be at least 1.")
        sys.exit(1)

    client = GPTImageClient(
        api_key=api_key,
        base_url=base_url,
        timeout=image_timeout,
    )

    image_count = 2
    output_dir = Path("output")
    output_dir.mkdir(parents=True, exist_ok=True)

    for index in range(image_count):
        print(
            f"Generating outline drawing {index + 1}/{image_count} "
            f"(timeout: {image_timeout}s)..."
        )
        images = client.generate_outline(
            description=description,
            model="gpt-image-2",
            size="1024x1024",
            n=1,
        )
        if not images:
            raise RuntimeError("Image API returned no image data.")
        path = images[0].save(output_dir / f"outline_{index:03d}.png")
        print(f"  Saved: {path}")
    print("Done.")


if __name__ == "__main__":
    main()
