import os
from pathlib import Path

import cv2
import numpy as np
import requests


ROOT = Path(__file__).resolve().parents[1]
GEMINI_MODEL = "gemini-2.5-flash"
GEMINI_ENDPOINT = (
    f"https://generativelanguage.googleapis.com/v1beta/models/"
    f"{GEMINI_MODEL}:generateContent"
)


def _load_api_key() -> str:
    """Load GEMINI_API_KEY from Streamlit secrets, env, or local .env."""
    try:
        import streamlit as st

        if "GEMINI_API_KEY" in st.secrets:
            return str(st.secrets["GEMINI_API_KEY"]).strip()
    except Exception:
        pass

    api_key = os.getenv("GEMINI_API_KEY", "").strip()
    if api_key:
        return api_key

    for env_path in (ROOT / ".env", Path(__file__).resolve().parent / ".env"):
        if not env_path.exists():
            continue
        for line in env_path.read_text(encoding="utf-8").splitlines():
            if not line or line.lstrip().startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            if key.strip() == "GEMINI_API_KEY":
                return value.strip().strip('"').strip("'")

    return ""


class GeminiTerrainAPI:
    """Gemini-backed terrain analysis plus deterministic local segmentation."""

    provider = "gemini-api"

    def __init__(self, api_key: str | None = None):
        self.api_key = (api_key or _load_api_key()).strip()

    @property
    def configured(self) -> bool:
        return bool(self.api_key)

    def segment_terrain(self, image_bgr: np.ndarray) -> np.ndarray:
        """
        Local pixel segmentation fallback.

        Gemini is used for the natural-language terrain brief; dense pixel masks
        stay local so the path planner remains fast and deterministic.
        """
        h, w = image_bgr.shape[:2]
        hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)

        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        seg = np.full((h, w), 4, dtype=np.uint8)

        green = (hue >= 35) & (hue <= 95) & (sat > 45) & (val > 35)
        sky = (hue >= 90) & (hue <= 125) & (sat > 35) & (val > 90)
        gray_rock = (sat < 45) & (val > 45) & (val < 190)
        dark_unknown = val < 35
        sand = ((hue <= 35) | (hue >= 170)) & (sat > 20) & (val > 80)
        lower_half = np.arange(h)[:, None] > (h * 0.45)

        seg[sand & lower_half] = 3
        seg[green] = 2
        seg[gray_rock] = 7
        seg[sky & ~lower_half] = 9
        seg[dark_unknown] = 0

        return seg

    def navigation_brief(self, metrics: dict) -> dict:
        """Call Gemini for a concise terrain/navigation summary."""
        if not self.configured:
            return {
                "used_api": False,
                "model": GEMINI_MODEL,
                "text": (
                    "Gemini API key not configured. Set GEMINI_API_KEY in .env "
                    "to generate the AI terrain brief."
                ),
            }

        prompt = (
            "You are assisting an off-road autonomous navigation demo. "
            "Give a concise 2-3 sentence terrain and route safety brief from "
            "these computed metrics. Do not invent details beyond the metrics.\n\n"
            f"Image size: {metrics['width']}x{metrics['height']} pixels\n"
            f"Navigable area: {metrics['navigable_percent']:.1f}%\n"
            f"Obstacle area: {metrics['obstacle_percent']:.1f}%\n"
            f"Path found: {metrics['path_found']}\n"
            f"Path length: {metrics['path_length']}\n"
            f"Segmentation backend: {metrics['inference_mode']}"
        )
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "maxOutputTokens": 120,
            },
        }

        try:
            response = requests.post(
                GEMINI_ENDPOINT,
                headers={
                    "x-goog-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json=payload,
                timeout=15,
            )
            response.raise_for_status()
            data = response.json()
            text = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            return {"used_api": True, "model": GEMINI_MODEL, "text": text}
        except Exception as exc:
            return {
                "used_api": False,
                "model": GEMINI_MODEL,
                "text": f"Gemini API request failed: {exc}",
            }
