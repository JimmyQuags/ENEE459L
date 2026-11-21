from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import binary_closing, binary_opening, gaussian_filter


def otsu_threshold(image: np.ndarray) -> int:
    """Return the Otsu threshold for a non-empty 2-D grayscale image."""
    grayscale = np.asarray(image)
    if grayscale.ndim != 2 or grayscale.size == 0:
        raise ValueError("image must be a non-empty two-dimensional array")
    if not np.issubdtype(grayscale.dtype, np.integer):
        raise ValueError("image pixel intensities must be integers")
    if np.any((grayscale < 0) | (grayscale > 255)):
        raise ValueError("image pixel intensities must be between 0 and 255")

    histogram = np.bincount(grayscale.ravel(), minlength=256).astype(np.float64)
    probabilities = histogram / grayscale.size
    levels = np.arange(256, dtype=np.float64)

    background_probability = np.cumsum(probabilities)
    background_sum = np.cumsum(levels * probabilities)
    foreground_probability = 1.0 - background_probability
    foreground_sum = background_sum[-1] - background_sum

    background_mean = np.divide(
        background_sum,
        background_probability,
        out=np.zeros_like(background_sum),
        where=background_probability > 0,
    )
    foreground_mean = np.divide(
        foreground_sum,
        foreground_probability,
        out=np.zeros_like(foreground_sum),
        where=foreground_probability > 0,
    )
    variance = (
        background_probability
        * foreground_probability
        * (background_mean - foreground_mean) ** 2
    )
    return int(np.argmax(variance))


def binarize_image(
    input_path: str | Path,
    output_path: str | Path,
    smoothing: float = 1.0,
    kernel_size: int = 3,
) -> int:
    """Binarize an image with Otsu thresholding and morphological cleanup."""
    if smoothing < 0:
        raise ValueError("smoothing must be non-negative")
    if not isinstance(kernel_size, (int, np.integer)) or kernel_size <= 0:
        raise ValueError("kernel_size must be a positive integer")

    with Image.open(input_path) as source:
        grayscale = np.asarray(source.convert("L"), dtype=np.uint8)

    if smoothing > 0:
        smoothed = gaussian_filter(grayscale.astype(np.float32), sigma=smoothing)
        grayscale = np.clip(np.rint(smoothed), 0, 255).astype(np.uint8)

    threshold = otsu_threshold(grayscale)
    mask = grayscale > threshold
    structure = np.ones((kernel_size, kernel_size), dtype=bool)
    cleaned = binary_closing(binary_opening(mask, structure=structure), structure=structure)

    result = (cleaned.astype(np.uint8) * 255)
    Image.fromarray(result, mode="L").save(output_path)
    return threshold


if __name__ == "__main__":
    image_path = Path(__file__).resolve().parent / "face.jpg"
    output_path = Path(__file__).resolve().parent / "otsu_binarized.png"
    threshold = binarize_image(image_path, output_path)
    print(f"Otsu threshold: {threshold}")
    print(f"Saved binary image to: {output_path}")
