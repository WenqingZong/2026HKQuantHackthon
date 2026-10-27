"""报告与产物落盘：markdown 表格格式、JSON 里的 numpy 标量。"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from hkq.report import markdown_table, write_json


def test_markdown_table_formats_cells() -> None:
    frame = pd.DataFrame({"count": [1234], "ic": [0.123456], "name": ["x"]}, index=["a"])
    frame.index.name = "row"

    assert markdown_table(frame).splitlines() == [
        "| row | count | ic | name |",
        "|---|---|---|---|",
        "| a | 1,234 | 0.1235 | x |",
    ]


def test_write_json_converts_numpy_scalars(tmp_path: Path) -> None:
    path = tmp_path / "metrics.json"

    write_json(path, {"n": np.int64(3), "ic": np.float64(0.5), "名称": "中文"})

    assert json.loads(path.read_text(encoding="utf-8")) == {"n": 3, "ic": 0.5, "名称": "中文"}
    assert "中文" in path.read_text(encoding="utf-8")


def test_write_json_rejects_unknown_types(tmp_path: Path) -> None:
    with pytest.raises(TypeError):
        write_json(tmp_path / "bad.json", {"value": object()})
