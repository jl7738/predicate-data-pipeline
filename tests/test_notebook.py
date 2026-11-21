"""The Colab notebook must stay valid nbformat and every code cell must at least compile."""
import ast
from pathlib import Path

import pytest

nbformat = pytest.importorskip("nbformat")
NB = Path(__file__).parent.parent / "notebooks" / "predx_colab_test.ipynb"


def test_notebook_is_valid_and_compiles():
    nb = nbformat.read(NB, 4)
    nbformat.validate(nb)
    for i, c in enumerate(nb.cells):
        if c.cell_type == "code":
            ast.parse(c.source, filename=f"cell{i}")
