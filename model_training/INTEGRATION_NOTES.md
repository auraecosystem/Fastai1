# LMLMxtend training integration notes

This is an integration-ready addition for `auraecosystem/Lmlmxtend`, scoped under
`model_training/lmlm_training/`. It deliberately does not modify the Rust
inference bridge or invent a concrete LMLM model constructor. The repository's
current Python training scripts contain separate legacy experiments; this
package provides a reusable native PyTorch training path without cloning or
fetching other repositories.

Copy `model_training/lmlm_training/` and `tests/test_lmlm_training.py` into the
LMLMxtend repository. Run:

```bash
python -m pytest -q tests/test_lmlm_training.py
python -m compileall -q model_training/lmlm_training
```

A real-model integration test still requires the actual LMLM model implementation,
its tokenizer/data contract, and a locally installed model/dataset. This package
is CPU-testable and does not claim GPU validation when CUDA/MPS hardware is absent.
