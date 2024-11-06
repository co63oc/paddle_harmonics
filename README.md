Code from <https://github.com/NVIDIA/torch-harmonics>

Install:

``` shell
# paddlepaddle develop
python -m pip install --pre paddlepaddle-gpu -i https://www.paddlepaddle.org.cn/packages/nightly/cu118/
pip install -r requirements.txt

pip install .

# test
pytest tests/

# example
mkdir examples/checkpoints
mkdir examples/figures
mkdir examples/output_data
python examples/train_sfno.py
```
