# LIBERO Benchmark

This example runs the LIBERO benchmark: https://github.com/Lifelong-Robot-Learning/LIBERO

This example requires git submodules to be initialized. Don't forget to run:

```bash
git submodule update --init --recursive
```

Terminal window 1:

```bash
# Create virtual environment
uv venv --python 3.9 examples/libero/.venv
source examples/libero/.venv/bin/activate
uv pip sync examples/libero/requirements.txt third_party/libero/requirements.txt  --extra-index-url https://download.pytorch.org/whl/cu121 --index-strategy=unsafe-best-match
uv pip install -e packages/openpi-client
uv pip install -e ../libero-env/LIBERO-original/.
uv pip install -r ../libero-env/LIBERO-original/libero_requirements.txt
uv pip install -e ../libero-env/LIBERO-plus/.
uv pip install -r ../libero-env/LIBERO-plus/libero_requirements.txt
export PYTHONPATH=$PYTHONPATH:$PWD/../libero-env/LIBERO-plus

# Run the simulation
python examples/libero/main.py
```

Terminal window 2:

```bash
# Run the server
uv run scripts/serve_policy.py --env LIBERO
```

> Notes: Detail running instructions can be found in the [README](../../README.md)