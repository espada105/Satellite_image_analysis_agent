from pathlib import Path

import cv2
import numpy as np
from fastapi.testclient import TestClient

from mcp_satellite_server.opencv_ops import ARTIFACT_DIR
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
            "ops": ["edges", "cloud_mask_like", "masking_like"],
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert len(data["ops"]) == 3
    assert data["ops"][0]["name"] == "edges"
    assert data["ops"][2]["name"] == "masking_like"
    for op in data["ops"]:
        assert op["artifact_uri"] is not None
        assert op["artifact_uri"].startswith("/imagery/artifacts/")
        artifact_name = op["artifact_uri"].split("/imagery/artifacts/", 1)[1]
        artifact_path = ARTIFACT_DIR / artifact_name
        assert artifact_path.exists()
