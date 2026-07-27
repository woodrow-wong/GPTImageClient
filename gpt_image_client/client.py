import os
import base64
import time
import mimetypes
from pathlib import Path
from typing import Optional, List, Literal, Union, Dict, Any
from dataclasses import dataclass
from urllib.parse import urlparse

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
        if self.api_key.startswith(("http://", "https://")):
            raise ValueError(
                "API key looks like a URL. Check that CODEX_API_KEY and "
                "CODEX_BASE_URL were not swapped."
            )

        parsed_base_url = urlparse(self.base_url)
        if parsed_base_url.scheme not in ("http", "https") or not parsed_base_url.netloc:
            hint = ""
            if self.base_url.startswith(("sk-", "key-")):
                hint = " It appears to contain an API key; rotate that key before continuing."
            raise ValueError(
                "Base URL must be a complete http:// or https:// URL. "
                "Check CODEX_BASE_URL and do not put an API key there." + hint
            )

    def _headers(self, content_type: str = "application/json") -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        if content_type:
            headers["Content-Type"] = content_type
        return headers

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
                last_error = f"Request timed out after {self.timeout} seconds"
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
        if n < 1:
            raise ValueError("n must be at least 1")

        payload = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": response_format,
        }
        if quality and quality != "standard":
            payload["quality"] = quality
        if style and style != "vivid":
            payload["style"] = style
        if user:
            payload["user"] = user

        # Some OpenAI-compatible proxies map this endpoint to the image tool,
        # which does not accept `n`. Request images individually for portability.
        images = []
        for _ in range(n):
            images.extend(
                self._request(
                    "POST",
                    "/v1/images/generations",
                    json_payload=payload,
                    max_retries=max_retries,
                )
            )
        return images

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

    def variations_batch(
        self,
        image_paths: List[Union[str, Path]],
        model: str = "gpt-image-2",
        n: int = 1,
        size: Literal["1024x1024", "256x256", "512x512"] = "1024x1024",
        response_format: Literal["url", "b64_json"] = "b64_json",
        user: Optional[str] = None,
        max_retries: int = 3,
    ) -> List[GeneratedImage]:
        """
        Create variations for multiple images in batch.

        Returns:
            List of GeneratedImage objects (all images from all source images).
        """
        all_results = []
        for path in image_paths:
            results = self.variation(
                image_path=path,
                model=model,
                n=n,
                size=size,
                response_format=response_format,
                user=user,
                max_retries=max_retries,
            )
            all_results.extend(results)
        return all_results

    def _encode_image(self, image_path: Union[str, Path]) -> str:
        path = Path(image_path)
        mime_type, _ = mimetypes.guess_type(str(path))
        if mime_type not in ("image/png", "image/jpeg", "image/jpg", "image/webp", "image/gif"):
            mime_type = "image/png"
        data = base64.b64encode(path.read_bytes()).decode("utf-8")
        return f"data:{mime_type};base64,{data}"

    def chat(
        self,
        messages: List[Dict[str, Any]],
        model: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: Optional[int] = None,
        max_retries: int = 3,
    ) -> str:
        """
        Send a chat completion request. Supports vision when message content
        includes image_url blocks.

        Args:
            messages: OpenAI-format chat messages. For images, include:
                {"role": "user", "content": [
                    {"type": "text", "text": "..."},
                    {"type": "image_url", "image_url": {"url": "data:image/png;base64,..."}}
                ]}
            model: Chat model name (e.g. gpt-4o).
            temperature: Sampling temperature.
            max_tokens: Max tokens in response.

        Returns:
            The assistant's text reply.
        """
        payload = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens

        url = f"{self.base_url}/v1/chat/completions"
        last_error = None

        for attempt in range(max_retries):
            try:
                resp = requests.post(
                    url,
                    headers=self._headers(),
                    json=payload,
                    timeout=self.timeout,
                )
                if resp.status_code == 401:
                    raise PermissionError(
                        f"Authentication failed (401). Check your CODEX_API_KEY: {resp.text}"
                    )
                if resp.status_code == 404:
                    raise ValueError(
                        f"Endpoint or model not found (404). Check base_url and model: {resp.text}"
                    )
                if resp.status_code == 429:
                    last_error = f"Rate limited (429)"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
                if not resp.ok:
                    last_error = f"HTTP {resp.status_code}: {resp.text}"
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue

                data = resp.json()
                return data["choices"][0]["message"]["content"]

            except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as e:
                last_error = str(e)
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)

        raise RuntimeError(
            f"Chat request failed after {max_retries} attempts. Last error: {last_error}"
        )

    def extract_outline_description(
        self,
        image_paths: List[Union[str, Path]],
        chat_model: str = "gpt-4o",
    ) -> str:
        """
        Analyze multiple images, extract common compositional elements,
        and return a description optimized for generating a line-art outline
        drawing (白描) suitable for gourd carving / traditional craft.

        Args:
            image_paths: List of image file paths to analyze together.
            chat_model: Vision-capable chat model to use.

        Returns:
            A text description of the common scene suitable for outline generation.
        """
        content_blocks = [
            {
                "type": "text",
                "text": (
                    "Please analyze these images together. They are different versions of the same scene. "
                    "Extract ONLY the elements that are COMMON across ALL images -- the shared composition, "
                    "core subjects, and essential spatial layout.\n\n"
                    "Ignore colors, textures, lighting, and details unique to individual versions.\n\n"
                    "Then output a SINGLE paragraph (in English) that describes this common scene "
                    "as a LINE ART / OUTLINE DRAWING (白描). The description must be optimized for "
                    "generating a black-ink outline drawing on white background, suitable for "
                    "traditional gourd carving art. Focus on:\n"
                    "- Silhouettes and contours, not shading\n"
                    "- Clear, simple lines\n"
                    "- Negative space\n"
                    "- Essential shapes only, no fine details\n"
                    "- Preserve the stable pose, relative placement, and major gesture lines\n"
                    "- Use any readable text only as thematic context; do not copy or reproduce text from the references\n"
                    "- Treat typography, borders, logos, captions, and graphic layout as irrelevant to the visual composition\n"
                    "- Treat glow, sparkles, fog, reflections, color, and photographic texture as irrelevant\n"
                    "- Prefer bold, connected contours that remain legible when carved on a small gourd\n\n"
                    "Output ONLY the description paragraph, nothing else."
                ),
            }
        ]
        for p in image_paths:
            content_blocks.append({
                "type": "image_url",
                "image_url": {"url": self._encode_image(p), "detail": "high"},
            })

        messages = [{"role": "user", "content": content_blocks}]
        return self.chat(messages, model=chat_model, temperature=0.5)

    def generate_outline(
        self,
        description: str,
        model: str = "gpt-image-2",
        size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        n: int = 1,
        max_retries: int = 1,
    ) -> List[GeneratedImage]:
        """
        Generate a line-art outline drawing from a scene description.

        Args:
            description: Scene description (e.g. from extract_outline_description()).
            model: Image generation model.
            size: Output image size.
            n: Number of images to generate.
            max_retries: Number of attempts per image. Defaults to one to avoid
                duplicate image jobs when a long-running request times out.

        Returns:
            List of GeneratedImage objects.
        """
        prompt = self._build_outline_prompt(description)
        return self.generate(
            prompt=prompt,
            model=model,
            n=n,
            size=size,
            response_format="b64_json",
            max_retries=max_retries,
        )

    def _build_outline_prompt(self, description: str) -> str:
        return (
            "Identity fidelity is the highest priority. Refine the preferred draft using the original Japanese anime "
            "character references; do not invent or redesign the character. Produce crisp black line art on pure white "
            "for later hand painting on a gourd. Follow the ordered specification below exactly. "
            f"{description}"
        )

    def generate_outline_from_references(
        self,
        image_paths: List[Union[str, Path]],
        description: str,
        model: str = "gpt-image-2",
        size: Literal["1024x1024", "1792x1024", "1024x1792"] = "1024x1024",
        n: int = 1,
        max_retries: int = 1,
    ) -> List[GeneratedImage]:
        """Generate identity-preserving line art from multiple character references."""
        if not image_paths:
            raise ValueError("At least one reference image is required")
        if n < 1:
            raise ValueError("n must be at least 1")

        paths = [Path(path) for path in image_paths]
        for path in paths:
            if not path.is_file():
                raise FileNotFoundError(f"Reference image not found: {path}")

        reference_instructions = (
            "The six attached images concern one original Japanese anime character. Reference 1 is the preferred "
            "line-art draft and controls only composition, pose, inscription placement, and general line-art density. "
            "References 2 and 3 override it for exact face identity, eyes, bangs, expression, hat, earrings, and anime "
            "style. Reference 4 overrides it for upper-body costume, right-facing head direction, staff, and open-palm "
            "flower. References 5 and 6 override it for full-body costume, hair, static stance, and water setting. Where "
            "the preferred draft conflicts with the original color references, always follow the color references. "
        )
        prompt = reference_instructions + self._build_outline_prompt(description)
        form_data = {
            "model": model,
            "prompt": prompt,
            "size": size,
            "response_format": "b64_json",
        }

        results = []
        for _ in range(n):
            open_files = []
            files = []
            try:
                for path in paths:
                    mime_type, _ = mimetypes.guess_type(str(path))
                    if not mime_type or not mime_type.startswith("image/"):
                        mime_type = "application/octet-stream"
                    file_obj = path.open("rb")
                    open_files.append(file_obj)
                    files.append(("image[]", (path.name, file_obj, mime_type)))

                results.extend(
                    self._request(
                        "POST",
                        "/v1/images/edits",
                        data=form_data,
                        files=files,
                        max_retries=max_retries,
                    )
                )
            finally:
                for file_obj in open_files:
                    file_obj.close()
        return results

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
