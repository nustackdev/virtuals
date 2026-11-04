"""The root README's Python blocks run as written, in order, in one namespace."""

from __future__ import annotations

import re
from pathlib import Path


README = Path(__file__).resolve().parents[2] / "README.md"


def test_readme_python_blocks_run(capsys):
    blocks = re.findall(r"```python\n(.*?)```", README.read_text(), re.S)
    assert len(blocks) >= 4, "README lost its code blocks"
    namespace: dict = {}
    for i, block in enumerate(blocks):
        exec(compile(block, f"README.md block {i}", "exec"), namespace)  # noqa: S102
    out = capsys.readouterr().out
    assert "alice: Alice" in out
    assert "bob: Bob" in out
