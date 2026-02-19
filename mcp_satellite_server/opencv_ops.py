from pathlib import Path
from urllib.request import urlopen

import cv2
import numpy as np

from mcp_satellite_server.schemas import OpResult

SUPPORTED_OPS = {"edges", "threshold", "morphology", "cloud_mask_like", "masking_like"}


def analyze_satellite_image(
    image_uri: str,
    ops: list[str],
    roi: dict | None = None,
) -> list[OpResult]:
    image = _load_image(image_uri)
    if image is None:
        raise ValueError(f"Failed to load image from {image_uri}")

    image = _apply_roi(image, roi)
    op_results: list[OpResult] = []

    for op in ops:
        if op not in SUPPORTED_OPS:
            op_results.append(OpResult(name=op, summary="unsupported op", stats={}))
            continue

        if op == "edges":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            edges = cv2.Canny(gray, 80, 160)
            ratio = float(np.count_nonzero(edges)) / float(edges.size)
            op_results.append(
                OpResult(
                    name=op,
                    summary=f"Edge density is {ratio:.2%}",
                    stats={"edge_density": round(ratio, 6)},
                )
            )
        elif op == "threshold":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 170, 255, cv2.THRESH_BINARY)
            bright_ratio = float(np.count_nonzero(binary)) / float(binary.size)
            op_results.append(
                OpResult(
                    name=op,
                    summary=f"Bright area ratio is {bright_ratio:.2%}",
                    stats={"bright_ratio": round(bright_ratio, 6)},
                )
            )
        elif op == "morphology":
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            _, binary = cv2.threshold(gray, 140, 255, cv2.THRESH_BINARY)
            kernel = np.ones((3, 3), np.uint8)
            morphed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)
            dense_ratio = float(np.count_nonzero(morphed)) / float(morphed.size)
            op_results.append(
                OpResult(
                    name=op,
                    summary=f"Morphology foreground ratio is {dense_ratio:.2%}",
                    stats={"foreground_ratio": round(dense_ratio, 6)},
                )
            )
        elif op == "cloud_mask_like":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower = np.array([0, 0, 180])
            upper = np.array([180, 80, 255])
            mask = cv2.inRange(hsv, lower, upper)
            cloud_ratio = float(np.count_nonzero(mask)) / float(mask.size)
            op_results.append(
                OpResult(
                    name=op,
                    summary=f"Estimated cloud-like coverage is {cloud_ratio:.2%}",
                    stats={"cloud_like_ratio": round(cloud_ratio, 6)},
                )
            )
        elif op == "masking_like":
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            lower = np.array([25, 30, 30])
            upper = np.array([95, 255, 255])
            mask = cv2.inRange(hsv, lower, upper)
            mask_ratio = float(np.count_nonzero(mask)) / float(mask.size)
            op_results.append(
                OpResult(
                    name=op,
                    summary=f"Simple color-mask coverage is {mask_ratio:.2%}",
                    stats={"mask_ratio": round(mask_ratio, 6)},
                )
            )

    return op_results


def _load_image(image_uri: str) -> np.ndarray | None:
    if image_uri.startswith("http://") or image_uri.startswith("https://"):
        with urlopen(image_uri) as response:  # nosec B310
            data = response.read()
        arr = np.asarray(bytearray(data), dtype=np.uint8)
        return cv2.imdecode(arr, cv2.IMREAD_COLOR)

    path = Path(image_uri)
    if not path.exists():
        return None
    return cv2.imread(str(path), cv2.IMREAD_COLOR)


def _apply_roi(image: np.ndarray, roi: dict | None) -> np.ndarray:
    if not roi:
        return image
    x = max(0, int(roi.get("x", 0)))
    y = max(0, int(roi.get("y", 0)))
    w = int(roi.get("w", image.shape[1]))
    h = int(roi.get("h", image.shape[0]))
    x2 = min(image.shape[1], x + max(w, 1))
    y2 = min(image.shape[0], y + max(h, 1))
    if x >= x2 or y >= y2:
        return image
    return image[y:y2, x:x2]
