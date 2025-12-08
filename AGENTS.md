# Repository Guidelines

This guide keeps VoxCPM contributions aligned and easy to review. Scan it before opening changes.

## Project Structure & Module Organization
- `src/voxcpm`: Library code. `core.py` wraps the model interface; `model/` holds VoxCPM graph logic; `modules/` contains LocDiT/MiniCPM-4/Audiovae blocks; `utils/` houses text normalization.
- `app.py`: Gradio UI for local demo. `streaming*.py` and `non_streaming.py` are runnable reference scripts.
- `examples/`: Sample text inputs. `data/` contains small prompt snippets; avoid growing it with heavy audio.
- `assets/`: Logos and docs visuals. `models/` may cache downloaded weights; do not assume it is tracked in CI.
- `results/`: Preferred place for evaluation notes or metrics. `.gitignore` already skips `*.wav` and `*.png`.

## Build, Test, and Development Commands
- Create env (example): `python -m venv .venv && source .venv/bin/activate` (or `.\.venv\Scripts\activate` on Windows).
- Install lib + extras: `pip install -e .` or `pip install -e .[dev]`.
- Run CLI locally: `voxcpm --text "Hello" --output out.wav` (uses HF cache if model not present).
- Launch web demo: `python app.py`.
- Try reference scripts: `python streaming.py`, `python streaming_cache.py`, `python non_streaming.py`.

## Coding Style & Naming Conventions
- Python 3.10+. Follow `black` (line length 120) and keep imports sorted/grouped.
- Use snake_case for functions/vars, PascalCase for classes, and explicit typing on public APIs.
- Prefer small, pure helpers; keep I/O paths configurable. Run `black .` and `flake8 src` before raising a PR.

## Testing Guidelines
- Framework: `pytest`. Run `pytest` or `pytest --cov=voxcpm` when adding logic.
- Place tests under `tests/` mirroring `src/voxcpm` modules. Use lightweight fixtures; reuse `data/` for small prompts and avoid bundling large audio.
- Include regression samples (text + expected tensor shapes/durations) instead of binary outputs.

## Commit & Pull Request Guidelines
- Commit messages: short, imperative summaries (e.g., `Add streaming cache helpers`). Current history is minimal—keep it clean.
- PRs should state goal, key changes, and test command outputs. Link issues when applicable and mention any model assets or new dependencies.
- Document new CLIs or config flags in `README.md` or `app.py` docstrings, and avoid committing large binaries; point reviewers to reproducible commands instead.
