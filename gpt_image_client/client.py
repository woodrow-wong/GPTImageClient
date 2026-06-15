import os
import base64
import time
from pathlib import Path
from typing import Optional, List, Literal, Union
from dataclasses import dataclass

import requests
from openai import OpenAI, APIError, APIConnectionError, RateLimitError, APITimeoutError


@dataclass
class GeneratedImage:
    url: Optional[str] = None
    b64_json: Optional[str] = None
    revised_prompt: Optional[str] = None

    def save(self, path: Union[str, Path]) -> Path:
        path = Path(path)
        if self.b64_json:
            data = base64.b64decode(self.b64_json)
            path.write_bytes(data)
        elif self.url:
            resp = requests.get(self.url, timeout=60)
            resp.raise_for_status()
            path.write_bytes(resp.content)
        else:
            raise ValueError("No image data (neither URL nor b64_json)")
        return path


def _to_generated_images(images) -> List[GeneratedImage]:
    return [
        GeneratedImage(
            url=getattr(img, "url", None),
            b64_json=getattr(img, "b64_json", None),
            revised_prompt=getattr(img, "revised_prompt", None),
        )
        for img in images
    ]


RETRYABLE_ERRORS = (APIConnectionError, RateLimitError, APITimeoutError)


class GPTImageClient:
    """
    Client for GPT Image-2 via Codex proxy, backed by the official OpenAI SDK.
    """

    DEFAULT_BASE_URL = "https://api.openai.com"

    def __init__(
        self,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        timeout: int = 120,
    ):
        self.api_key = api_key or os.environ.get("CODEX_API_KEY", "")
        self.base_url = (base_url or os.environ.get("CODEX_BASE_URL", self.DEFAULT_BASE_URL)).rstrip("/")
        self.timeout = timeout

        if not self.api_key:
            raise ValueError(
                "API key is required. Set CODEX_API_KEY env var or pass api_key parameter."
            )

        self._client = OpenAI(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
            max_retries=0,  # we handle retries ourselves for fine-grained control
        )

    def generate(
        self,
        prompt: str,
        model: str = "gpt-image-2",
        n: int = 1,
        size: Literal["1024x1024", "1792x1024", "1024x1792", "256x256", "512x512"] = "1024x1024",
        quality: Literal["standard", "hd"] = "standard",
        style: Literal["vivid", "natural"] = "vivid",
        response_format: Literal["url", "b64_json"] = "b64_json",
        user: Optional[str] = None,
        max_retries: int = 3,
    ) -> List[GeneratedImage]:
        """
        Generate images from a text prompt via OpenAI SDK images.generate().

        Returns:
            List of GeneratedImage objects.
        """
        kwargs = dict(
            model=model,
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
        )
        if model == "gpt-image-2" or "dall-e-3" in model:
            kwargs["quality"] = quality
            kwargs["style"] = style
        if user:
            kwargs["user"] = user

        return self._retry(
            lambda: self._client.images.generate(**kwargs),
            max_retries,
            extractor=lambda r: _to_generated_images(r.data),
        )

    def edit(
        self,
        image_path: Union[str, Path],
        prompt: str,
        mask_path: Optional[Union[str, Path]] = None,
        model: str = "gpt-image-2",
        n: int = 1,
        size: Literal["1024x1024", "256x256", "512x512"] = "1024x1024",
        response_format: Literal["url", "b64_json"] = "b64_json",
        user: Optional[str] = None,
        max_retries: int = 3,
    ) -> List[GeneratedImage]:
        """
        Edit an existing image via OpenAI SDK images.edit().

        Returns:
            List of GeneratedImage objects.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        kwargs = dict(
            model=model,
            image=image_path.open("rb"),
            prompt=prompt,
            n=n,
            size=size,
            response_format=response_format,
        )
        if mask_path:
            mask_path = Path(mask_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            kwargs["mask"] = mask_path.open("rb")
        if user:
            kwargs["user"] = user

        try:
            return self._retry(
                lambda: self._client.images.edit(**kwargs),
                max_retries,
                extractor=lambda r: _to_generated_images(r.data),
            )
        finally:
            for f in (kwargs.get("image"), kwargs.get("mask")):
                if f:
                    f.close()

    def variation(
        self,
        image_path: Union[str, Path],
        model: str = "gpt-image-2",
        n: int = 1,
        size: Literal["1024x1024", "256x256", "512x512"] = "1024x1024",
        response_format: Literal["url", "b64_json"] = "b64_json",
        user: Optional[str] = None,
        max_retries: int = 3,
    ) -> List[GeneratedImage]:
        """
        Create variations via OpenAI SDK images.create_variation().

        Returns:
            List of GeneratedImage objects.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        kwargs = dict(
            model=model,
            image=image_path.open("rb"),
            n=n,
            size=size,
            response_format=response_format,
        )
        if user:
            kwargs["user"] = user

        try:
            return self._retry(
                lambda: self._client.images.create_variation(**kwargs),
                max_retries,
                extractor=lambda r: _to_generated_images(r.data),
            )
        finally:
            kwargs["image"].close()

    def _retry(self, fn, max_retries: int, extractor=None):
        last_error = None
        for attempt in range(max_retries):
            try:
                result = fn()
                return extractor(result) if extractor else result
            except RETRYABLE_ERRORS as e:
                last_error = e
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except APIError as e:
                raise RuntimeError(f"API error: {e}") from e
        raise RuntimeError(
            f"Request failed after {max_retries} attempts: {last_error}"
        )

    def save_all(
        self,
        images: List[GeneratedImage],
        output_dir: Union[str, Path] = "./output",
        prefix: str = "image",
    ) -> List[Path]:
        """
        Save all generated images to a directory.

        Args:
            images: List of GeneratedImage from generate/edit/variation.
            output_dir: Directory to save images.
            prefix: Filename prefix.

        Returns:
            List of saved file paths.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            ext = ".png"
            fname = f"{prefix}_{i:03d}{ext}"
            path = img.save(output_dir / fname)
            paths.append(path)
        return paths
