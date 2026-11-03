"""Vision / OCR / screenshot / clipboard tools. Optional deps: Pillow, rapidocr-onnxruntime, pyperclip."""
from __future__ import annotations

import base64
import io
import mimetypes
import tempfile
import time
from pathlib import Path

from ..client import DEFAULT_VISION_MODEL, NIMError
from ..core import Registry
from ..util import have


def _encode_image(p: Path, limit_b64: int = 170_000) -> str:
    """Return a data: URI. Downscales with Pillow so hosted NIM VLM endpoints accept it."""
    raw = p.read_bytes()
    mime = mimetypes.guess_type(p.name)[0] or "image/png"
    if len(base64.b64encode(raw)) <= limit_b64:
        return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())
    if not have("PIL"):
        return "data:%s;base64,%s" % (mime, base64.b64encode(raw).decode())  # let the API decide
    from PIL import Image
    img = Image.open(io.BytesIO(raw)).convert("RGB")
    size, quality = 1280, 85
    while True:
        im = img.copy()
        im.thumbnail((size, size))
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=quality)
        b64 = base64.b64encode(buf.getvalue())
        if len(b64) <= limit_b64 or size <= 256:
            return "data:image/jpeg;base64," + b64.decode()
        size, quality = int(size * 0.8), max(40, quality - 8)


def register(reg: Registry) -> None:
    @reg.tool("analyze_image",
              "Look at an image (local path or http(s) URL) with a NVIDIA NIM vision model and answer a question about it "
              "(screenshots, diagrams, photos, UI mockups, charts).",
              {"source": {"type": "string", "description": "image path or URL"},
               "question": {"type": "string", "description": "what to look for (default: describe in detail)"}},
              ["source"], group="media", read_only=True, summary_key="source")
    def analyze_image(ctx, source, question="Describe this image in detail, transcribing any visible text."):
        if source.startswith(("http://", "https://")):
            url = source
        else:
            p = ctx.path(source)
            if not p.is_file():
                return "Error: image not found: %s" % p
            url = _encode_image(p)
        model = ctx.config.get("vision_model") or DEFAULT_VISION_MODEL
        msgs = [{"role": "user", "content": [{"type": "text", "text": question},
                                              {"type": "image_url", "image_url": {"url": url}}]}]
        try:
            r = ctx.client.chat_stream(model, msgs, tools=None, max_tokens=1500, temperature=0.2)
        except NIMError as e:
            return ("Error from vision model '%s': %s. Pick a vision-capable model with /vision-model <id> "
                    "(see /models)." % (model, e.message))
        return r["content"].strip() or "(vision model returned nothing)"

    @reg.tool("ocr_image", "Extract text from an image locally with OCR (needs `pip install rapidocr-onnxruntime`).",
              {"path": {"type": "string"}}, ["path"], group="media", read_only=True, summary_key="path")
    def ocr_image(ctx, path):
        p = ctx.path(path)
        if not p.is_file():
            return "Error: image not found: %s" % p
        try:
            from rapidocr_onnxruntime import RapidOCR
        except ImportError:
            return "Error: local OCR needs `pip install rapidocr-onnxruntime`. Alternatively use analyze_image."
        result, _ = RapidOCR()(str(p))
        if not result:
            return "(no text found)"
        return "\n".join(r[1] for r in result)

    @reg.tool("screenshot", "Capture the screen to a PNG file (Windows/macOS; Linux needs a display). Then use analyze_image on it.",
              {"path": {"type": "string", "description": "output path (optional)"}}, [], group="media", ask=True)
    def screenshot(ctx, path=""):
        try:
            from PIL import ImageGrab
        except ImportError:
            return "Error: screenshots need `pip install pillow`."
        out = ctx.path(path) if path else Path(tempfile.gettempdir()) / ("nimcode_shot_%d.png" % int(time.time()))
        try:
            ImageGrab.grab().save(out)
        except Exception as e:
            return "Error: could not capture the screen: %s" % e
        return "Saved screenshot to %s" % out

    @reg.tool("clipboard_read", "Read text from the system clipboard (needs `pip install pyperclip`).", {}, [],
              group="media", ask=True)
    def clipboard_read(ctx):
        try:
            import pyperclip
            return pyperclip.paste() or "(clipboard is empty)"
        except ImportError:
            return "Error: needs `pip install pyperclip`."
        except Exception as e:
            return "Error: %s" % e

    @reg.tool("clipboard_write", "Copy text to the system clipboard (needs `pip install pyperclip`).",
              {"text": {"type": "string"}}, ["text"], group="media", ask=True, summary_key="text")
    def clipboard_write(ctx, text):
        try:
            import pyperclip
            pyperclip.copy(text)
            return "Copied %d characters to the clipboard." % len(text)
        except ImportError:
            return "Error: needs `pip install pyperclip`."
        except Exception as e:
            return "Error: %s" % e
