python -m pytest -q tests/test_lmlm_training.py
python -m compileall -q model_training/lmlm_training
python -m pip install -e ".[training,dev]"
