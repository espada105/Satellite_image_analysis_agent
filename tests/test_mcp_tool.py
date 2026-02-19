from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from mcp_satellite_server.server import app


def test_mcp_analyze_satellite_image(tmp_path: Path) -> None:
    image_path = tmp_path / "sample.png"
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[10:60, 10:60] = (255, 255, 255)
    cv2.imwrite(str(image_path), image)

    client = TestClient(app)
    response = client.post(
        "/tools/analyze_satellite_image",
        json={
            "image_uri": str(image_path),
            "ops": ["edges", "cloud_mask_like"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["ops"]) == 2
    assert data["ops"][0]["name"] == "edges"
