import ast
from pathlib import Path

PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


def test_streamlit_does_not_import_agent_directly():
    """Streamlit入口不能绕过FastAPI直接依赖Agent。"""
    app_path = PROJECT_ROOT / "app.py"
    source = app_path.read_text(
        encoding="utf-8"
    )
    syntax_tree = ast.parse(source)

    imported_modules: set[str] = set()

    for node in ast.walk(
        syntax_tree
    ):
        if isinstance(
            node,
            ast.Import,
        ):
            imported_modules.update(
                alias.name
                for alias in node.names
            )
        elif isinstance(
            node,
            ast.ImportFrom,
        ):
            if node.module:
                imported_modules.add(
                    node.module
                )

    assert "agent.react_agent" not in (
        imported_modules
    )
    assert "frontend.api_client" in (
        imported_modules
    )
