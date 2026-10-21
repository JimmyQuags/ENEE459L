from __future__ import annotations

import math
from numbers import Integral, Real
from typing import Any, Iterable, Sequence

from schema import BLACK_MAX, DARK_MEAN, FLAT_SPREAD

import cv2
import time
import json
import numpy as np
from pathlib import Path

class FrameError(ValueError):
    """A shape or a format this module cannot price. `capture.py` exits 3."""


# The formats this lab prices, in the order Lecture 07 slide 27 lists them.
# Written as a tuple rather than derived from the function below, because
# `_bytes_per_pixel` is stubbed in the student tree and a starter whose
# FORMATS list was empty until the stub was filled would fail in a way that
# points at the wrong line.
FORMATS: tuple[str, ...] = ("raw10", "gray8", "nv12", "i420", "rgb", "bgr",
                            "bgrx")

# What each format is, in one clause, for the summary table and the error
# message. Not used by any arithmetic.
FORMAT_NOTES: dict[str, str] = {
    "raw10": "on the CSI wire, four pixels packed into five bytes, no ISP yet",
    "gray8": "luma only, which is a decision about colour and not a format",
    "nv12": "luma full, chroma quartered and interleaved -- the ISP's output",
    "i420": "the same bytes as NV12 with the two chroma planes kept apart",
    "rgb": "three bytes, nothing shared, nothing wasted",
    "bgr": "the same three bytes, ordered the way OpenCV expects them",
    "bgrx": "four bytes, one of which nobody reads, bought for alignment",
}


def _bytes_per_pixel(fmt: str) -> float:
    table = {
        # One byte of luma per pixel, plus a quarter of a byte of Cb and a
        # quarter of a byte of Cr, because the chroma plane is half width and
        # half height. NV12 interleaves the two chroma channels in one plane
        # and I420 keeps them in two, which changes the layout and not the
        # total, which is why both entries are the same number.
        "nv12": 1.5,
        "i420": 1.5,
        # Ten bits per pixel on the wire: four pixels in five bytes. This is
        # the only entry describing something that has not reached memory yet.
        "raw10": 1.25,
        "gray8": 1.0,
        "rgb": 3.0,
        "bgr": 3.0,
        # The fourth byte is padding. It carries no information and it costs
        # exactly as much to move as the three that do.
        "bgrx": 4.0,
    }
    if not isinstance(fmt, str):
        raise FrameError(f"format must be a string, got {type(fmt).__name__}")
    key = fmt.lower()
    if key not in table:
        raise FrameError(
            f"format {fmt!r} is not one of {', '.join(FORMATS)}. If you meant "
            "a format this lab does not price, add it here and say in your "
            "paragraph what its bytes per pixel is and how you know."
        )
    return table[key]


def _validate_dimensions(width: int, height: int) -> None:
    if (isinstance(width, bool) or not isinstance(width, Integral) or width <= 0
            or isinstance(height, bool) or not isinstance(height, Integral)
            or height <= 0):
        raise FrameError("width and height must be positive integers")


def _validate_format_dimensions(width: int, height: int, fmt: str) -> None:
    _validate_dimensions(width, height)
    _bytes_per_pixel(fmt)


# ---------------------------------------------------------------------------
# Hardware Capture & Pipeline Configuration
# ---------------------------------------------------------------------------

def gstreamer_pipeline(
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 1920,
    display_height: int = 1080,
    framerate: int = 30,
    flip_method: int = 0,
) -> str:
    return (
        f"nvarguscamerasrc ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, "
        f"format=(string)NV12, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=1"
    )


def grab_frame(pipeline: str, warmup_frames: int = 15) -> np.ndarray:
    cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not cap.isOpened():
        raise RuntimeError("Failed to open camera via GStreamer pipeline.")

    try:
        # Warmup loop to allow sensor auto-exposure and AWB to converge
        for _ in range(warmup_frames):
            ret, _ = cap.read()
            if not ret:
                time.sleep(0.03)

        ret, frame = cap.read()
        if not ret or frame is None:
            raise RuntimeError("Camera opened, but failed to capture a valid frame.")
        return frame
    finally:
        cap.release()


############################
## Your work starts here ###
############################

def frame_bytes(width: int, height: int, fmt: str) -> int:
    _validate_format_dimensions(width, height, fmt)
    return math.ceil(width * height * _bytes_per_pixel(fmt))


def bytes_per_second(width: int, height: int, fmt: str, fps: float) -> float:
    if (isinstance(fps, bool) or not isinstance(fps, Real)
            or not math.isfinite(fps) or fps <= 0):
        raise FrameError("fps must be a finite, positive number")
    return frame_bytes(width, height, fmt) * fps


def frame_stats(pixels: Sequence[float] | Iterable[float]) -> dict[str, float]:
    values = np.asarray(list(pixels), dtype=np.float64)
    if values.ndim != 1 or values.size == 0:
        raise FrameError("pixels must be a non-empty, one-dimensional iterable")
    if not np.all(np.isfinite(values)):
        raise FrameError("pixels must contain only finite values")
    if np.any((values < 0.0) | (values > 1.0)):
        raise FrameError("pixels must be normalized to the range [0, 1]")
    minimum = float(np.min(values))
    maximum = float(np.max(values))
    return {
        "min": minimum,
        "max": maximum,
        "mean": float(np.mean(values)),
        "spread": maximum - minimum,
        "sd": float(np.std(values)),
        "samples": float(values.size),
    }


def classify(stats: dict[str, float]) -> tuple[str, str]:
    required = ("max", "spread", "mean")
    if any(key not in stats for key in required):
        raise FrameError(f"stats must contain {', '.join(required)}")
    values = {key: stats[key] for key in required}
    if any(isinstance(value, bool) or not isinstance(value, Real)
           or not math.isfinite(value) for value in values.values()):
        raise FrameError("stats values must be finite numbers")
    if not (0.0 <= values["max"] <= 1.0 and 0.0 <= values["mean"] <= 1.0):
        raise FrameError("stats max and mean must be normalized to [0, 1]")
    if values["spread"] < 0.0:
        raise FrameError("stats spread cannot be negative")

    mean = values["mean"]
    spread = values["spread"]
    if values["max"] <= BLACK_MAX:
        return "black", f"max {values['max']:.4f} is at or below the black threshold."
    if spread < FLAT_SPREAD:
        return "flat", f"spread {spread:.4f} is below the structure threshold."
    if mean < DARK_MEAN:
        return (
            "dark",
            f"mean {mean:.4f} is below {DARK_MEAN:.4f}, "
            "but the frame has structure.",
        )
    return "scene", f"mean {mean:.4f}, spread {spread:.4f}. Exposed and structured."


def conversion_bytes(width: int, height: int, src: str, dst: str) -> int:
    if not isinstance(src, str) or not isinstance(dst, str):
        raise FrameError("source and destination formats must be strings")
    if src.lower() == dst.lower():
        return 0
    return frame_bytes(width, height, src) + frame_bytes(width, height, dst)


def resize_bytes(src_w: int, src_h: int, dst_w: int, dst_h: int,
                 fmt: str) -> int:
    _validate_format_dimensions(src_w, src_h, fmt)
    _validate_format_dimensions(dst_w, dst_h, fmt)
    return frame_bytes(src_w, src_h, fmt) + frame_bytes(dst_w, dst_h, fmt)


def normalise_bytes(width: int, height: int, channels: int = 3,
                    out_bytes: int = 4) -> int:
    _validate_dimensions(width, height)
    for name, value in (("channels", channels), ("out_bytes", out_bytes)):
        if isinstance(value, bool) or not isinstance(value, Integral) or value <= 0:
            raise FrameError(f"{name} must be a positive integer")
    pixels = width * height * channels
    return pixels + pixels * out_bytes


def summarise_formats(width: int, height: int) -> list[dict[str, Any]]:
    rgb_bytes = frame_bytes(width, height, "rgb")
    return [
        {
            "format": fmt,
            "bytes_per_pixel": _bytes_per_pixel(fmt),
            "bytes": frame_bytes(width, height, fmt),
            "relative_to_rgb": frame_bytes(width, height, fmt) / rgb_bytes,
            "note": FORMAT_NOTES[fmt],
        }
        for fmt in FORMATS
    ]


###############################
##### The main function #######
###############################
def main(
    output_dir: str = "output",
    image_filename: str = "captured_frame.jpg",
    json_filename: str = "frame_report.json",
    target_fps: float = 30.0,
) -> None:
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    img_path = out_path / image_filename
    json_path = out_path / json_filename

    # 1. Capture frame from hardware
    pipeline_str = gstreamer_pipeline()
    frame_bgr = grab_frame(pipeline_str)

    # 2. Save captured image to disk
    cv2.imwrite(str(img_path), frame_bgr)
    height, width, _ = frame_bgr.shape

    # 3. Extract normalized luma/grayscale values in [0.0, 1.0] for classification
    # classify() tests against BLACK_MAX (1/255 ≈ 0.0039) and mean < 0.10
    gray_frame = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
    normalized_luma = gray_frame.astype(np.float32).ravel() / 255.0

    # 4. Compute statistics and scene category
    stats = frame_stats(normalized_luma)
    classification, reason = classify(stats)

    # 5. Calculate memory bus bandwidth and conversion costs
    formats_table = summarise_formats(width, height)
    bgr_bandwidth_bps = bytes_per_second(width, height, "bgr", target_fps)
    nv12_to_bgr_cost = conversion_bytes(width, height, "nv12", "bgr")
    norm_cost = normalise_bytes(width, height, channels=3, out_bytes=4)

    # Downsampling budget example (1080p -> 224x224 input tensor)
    resize_cost = resize_bytes(width, height, 224, 224, "rgb")

    # 6. Aggregate payload and write JSON
    payload: dict[str, Any] = {
        "capture_metadata": {
            "saved_image_path": str(img_path.resolve()),
            "width": width,
            "height": height,
            "channels": 3,
            "assumed_fps": target_fps,
        },
        "frame_analysis": {
            "classification": classification,
            "classification_reason": reason,
            "stats": stats,
        },
        "memory_pricing": {
            "bandwidth_bytes_per_sec": {
                "bgr_live_stream": bgr_bandwidth_bps,
                "nv12_live_stream": bytes_per_second(width, height, "nv12", target_fps),
            },
            "format_comparison": formats_table,
            "pipeline_stage_costs": {
                "nv12_to_bgr_conversion_bytes": nv12_to_bgr_cost,
                "resize_to_224x224_rgb_bytes": resize_cost,
                "fp32_normalization_tensor_bytes": norm_cost,
            },
        },
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=4)

    print(f"Captured frame saved to : {img_path}")
    print(f"Analysis saved to       : {json_path}")
    print(f"Classification result   : {classification.upper()} ({reason})")

if __name__ == "__main__":
    main()
    