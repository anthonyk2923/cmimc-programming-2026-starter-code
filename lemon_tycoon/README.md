# Lemon Tycoon

## Local setup

If your system Python is "externally managed" (PEP 668), install dependencies in a virtual environment:

```bash
cd lemon_tycoon
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run locally

Edit `submission.py`, then:

```bash
python run.py          # simulate a game
python visualize.py    # animated playback
```

Change `PLAYER_CTORS` in either file to mix opponents.
