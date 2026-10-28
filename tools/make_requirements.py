"""Write requirements files using the versions installed in YOUR venv.

Run from the project root (venv active):
    python tools/make_requirements.py
"""
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

API_PACKAGES = ["fastapi", "uvicorn", "pydantic", "sqlalchemy",
                "scikit-learn", "pandas", "numpy", "joblib"]
DASHBOARD_PACKAGES = ["streamlit", "requests", "pandas"]


def write(path: Path, packages: list[str]):
    lines, missing = [], []
    for name in packages:
        try:
            lines.append(f"{name}=={version(name)}")
        except PackageNotFoundError:
            missing.append(name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")
    print(f"\n{path.relative_to(ROOT)}:")
    print("\n".join("  " + l for l in lines))
    if missing:
        print("  NOT INSTALLED (skipped):", ", ".join(missing))


write(ROOT / "requirements.txt", API_PACKAGES)
write(ROOT / "dashboard" / "requirements.txt", DASHBOARD_PACKAGES)