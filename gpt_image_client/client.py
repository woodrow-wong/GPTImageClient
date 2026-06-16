import os
import base64
import time
from pathlib import Path
from typing import Optional, List, Literal, Union
from dataclasses import dataclass

import requests


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


class GPTImageClient:
    """
    Client for GPT Image-2 via Codex proxy, using raw HTTP requests
    for maximum compatibility with proxy/relay services.
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

    def _headers(self, content_type: str = "application/json") -> dict:
        return {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": content_type,
        }

    def _parse_response(self, data: dict) -> List[GeneratedImage]:
        return [
            GeneratedImage(
                url=item.get("url"),
                b64_json=item.get("b64_json"),
                revised_prompt=item.get("revised_prompt"),
            )
            for item in data.get("data", [])
        ]

    def _request(self, method: str, path: str, json_payload: dict = None,
                 files: dict = None, data: dict = None, max_retries: int = 3) -> List[GeneratedImage]:
        url = f"{self.base_url}{path}"
        last_error = None

        for attempt in range(max_retries):
            try:
                if files:
                    resp = requests.request(
                        method, url,
                        headers=self._headers(None),
                        data=data,
                        files=files,
                        timeout=self.timeout,
                    )
                else:
                    resp = requests.request(
                        method, url,
                        headers=self._headers(),
                        json=json_payload,
                        timeout=self.timeout,
                    )

                if resp.status_code == 401:
                    raise PermissionError(
                        f"Authentication failed (401). Check your CODEX_API_KEY. "
                        f"Response: {resp.text}"
                    )
                if resp.status_code == 404:
                    raise ValueError(
                        f"Endpoint or model not found (404). "
                        f"Check CODEX_BASE_URL and model name. Response: {resp.text}"
                    )
                if resp.status_code == 429:
                    last_error = f"Rate limited (429): {resp.text}"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
                if not resp.ok:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue

                return self._parse_response(resp.json())

            except requests.exceptions.Timeout:
                last_error = "Request timed out"
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)
            except requests.exceptions.ConnectionError as e:
                last_error = f"Connection error: {e}"
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Request failed after {max_retries} attempts. Last error: {last_error}"
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
        Generate images from a text prompt.

        Returns:
            List of GeneratedImage objects.
        """
        payload = {
            "model": model,
            "prompt": prompt,
            "n": n,
            "size": size,
            "response_format": response_format,
        }
        if quality and quality != "standard":
            payload["quality"] = quality
        if style and style != "vivid":
            payload["style"] = style
        if user:
            payload["user"] = user

        return self._request("POST", "/v1/images/generations", json_payload=payload, max_retries=max_retries)

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
        Edit an existing image based on a prompt.
        Image must be a square PNG, less than 4MB.

        Returns:
            List of GeneratedImage objects.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        form_data = {
            "model": model,
            "prompt": prompt,
            "n": str(n),
            "size": size,
            "response_format": response_format,
        }
        if user:
            form_data["user"] = user

        files = {
            "image": (image_path.name, image_path.open("rb"), "image/png"),
        }
        if mask_path:
            mask_path = Path(mask_path)
            if not mask_path.is_file():
                raise FileNotFoundError(f"Mask file not found: {mask_path}")
            files["mask"] = (mask_path.name, mask_path.open("rb"), "image/png")

        try:
            return self._request(
                "POST", "/v1/images/edits",
                data=form_data, files=files,
                max_retries=max_retries,
            )
        finally:
            for _, (_, f, _) in files.items():
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
        Create variations of an existing image.
        Image must be a square PNG, less than 4MB.

        Returns:
            List of GeneratedImage objects.
        """
        image_path = Path(image_path)
        if not image_path.is_file():
            raise FileNotFoundError(f"Image file not found: {image_path}")

        form_data = {
            "model": model,
            "n": str(n),
            "size": size,
            "response_format": response_format,
        }
        if user:
            form_data["user"] = user

        files = {
            "image": (image_path.name, image_path.open("rb"), "image/png"),
        }

        try:
            return self._request(
                "POST", "/v1/images/variations",
                data=form_data, files=files,
                max_retries=max_retries,
            )
        finally:
            files["image"][1].close()

    def save_all(
        self,
        images: List[GeneratedImage],
        output_dir: Union[str, Path] = "./output",
        prefix: str = "image",
    ) -> List[Path]:
        """
        Save all generated images to a directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        paths = []
        for i, img in enumerate(images):
            fname = f"{prefix}_{i:03d}.png"
            path = img.save(output_dir / fname)
            paths.append(path)
        return paths
