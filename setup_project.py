"""Lab 01 - Task 1: AI project directory structure generator."""
from pathlib import Path

FOLDERS = ["data/raw", "data/processed", "models", "notebooks", "src", "configs", "logs"]

GITIGNORE = """# ---- Python ----
__pycache__/
*.py[cod]
*.egg-info/
build/
dist/

# ---- Virtual environments ----
.venv/
venv/
aipdd_env/
env/

# ---- Jupyter ----
.ipynb_checkpoints/

# ---- Binary model weights ----
*.pt
*.onnx
*.h5

# ---- Logs / OS / IDE ----
logs/*.log
.DS_Store
Thumbs.db
.vscode/
.idea/
"""

for folder in FOLDERS:
    p = Path(folder)
    p.mkdir(parents=True, exist_ok=True)
    (p / ".gitkeep").touch()
Path(".gitignore").write_text(GITIGNORE)
print("Project structure created!")
