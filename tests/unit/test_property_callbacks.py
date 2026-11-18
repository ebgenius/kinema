"""Property callbacks Blender can call: no lambdas in deferred annotations.

In a module with ``from __future__ import annotations``, a property declared
in a class annotation is kept as a string, and Blender evaluates it when the
class is registered. A name in it is looked up then: ``update=module.func``
finds ``module``. A lambda's body is looked up later, when Blender calls it,
in the globals of that evaluation, which are not the module's. So a poll
written ``lambda self, obj: references.may_be_reference(self, obj)`` raised
``NameError`` on every object the Relative To field's search offered, and the
search found nothing. Calling the function from Python, as a test does,
never goes that way.
"""

from __future__ import annotations

import ast
from pathlib import Path

SOURCE = Path(__file__).resolve().parents[2] / "src" / "kinema"


def _deferred(tree: ast.Module) -> bool:
    return any(
        isinstance(node, ast.ImportFrom)
        and node.module == "__future__"
        and any(alias.name == "annotations" for alias in node.names)
        for node in tree.body
    )


def _lambdas_in_annotations(tree: ast.Module):
    for cls in ast.walk(tree):
        if not isinstance(cls, ast.ClassDef):
            continue
        for statement in cls.body:
            if not isinstance(statement, ast.AnnAssign):
                continue
            for node in ast.walk(statement.annotation):
                if isinstance(node, ast.Lambda):
                    yield f"{cls.name}.{ast.unparse(statement.target)} (line {node.lineno})"


def test_no_property_annotation_holds_a_lambda_where_annotations_are_deferred():
    found = []
    for path in sorted(SOURCE.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        if _deferred(tree):
            where = _lambdas_in_annotations(tree)
            found += [f"{path.relative_to(SOURCE)}: {place}" for place in where]
    assert not found, "pass a module-level function instead: " + "; ".join(found)


def test_the_check_finds_one():
    """The check itself, on the declaration as it was."""
    tree = ast.parse(
        "from __future__ import annotations\n"
        "class KinemaWaypoint:\n"
        "    reference: PointerProperty(\n"
        "        poll=lambda self, obj: references.may_be_reference(self, obj))\n"
    )
    assert _deferred(tree)
    assert list(_lambdas_in_annotations(tree)) == ["KinemaWaypoint.reference (line 4)"]
