Install the docs dependencies and serve with:

```powershell
# from the workspace root
uv run --group docs mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser. MkDocs watches for file changes and live-reloads automatically.

To build a static output instead (into `site/`):
```powershell
uv run --group docs mkdocs build
```