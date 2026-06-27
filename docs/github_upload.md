# GitHub Upload Guide

## Files To Commit

Commit source code, documentation, configuration and tests:

```text
.github/
configs/
docs/
scripts/
src/
tests/
.gitignore
LICENSE
README.md
pyproject.toml
requirements.txt
```

Do not commit:

```text
.venv/
.idea/
data/
outputs/
checkpoints/
*.pt
*.onnx
*.egg-info/
```

These are already covered by `.gitignore`.

## Suggested First Commit

```powershell
git add .github configs docs scripts src tests .gitignore LICENSE README.md pyproject.toml requirements.txt
git commit -m "Initial lane binary classifier project"
```

## Connect To A New GitHub Repository

After creating an empty repository on GitHub:

```powershell
git branch -M main
git remote add origin https://github.com/<your-name>/<your-repo>.git
git push -u origin main
```

Replace `<your-name>` and `<your-repo>` with your actual GitHub account and repository name.

## Recommended Repository Description

```text
Lightweight lane in/out binary classifier with TensorBoard, ONNX export, and ONNX Runtime inference.
```
