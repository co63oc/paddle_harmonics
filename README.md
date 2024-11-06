# paddle_harmonics(Paddle Backend)

> [!IMPORTANT]
> This branch(paddle) experimentally supports [Paddle backend](https://www.paddlepaddle.org.cn/en/install/quick?docurl=/documentation/docs/en/develop/install/pip/linux-pip_en.html)
> as almost all the core code has been completely rewritten using the Paddle API.
>
> It is recommended to install **nightly-build(develop)** Paddle before running any code in this branch.

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

# notebooks
mkdir notebooks/data
mkdir notebooks/plots
chmod a+rwx notebooks/data
chmod a+rwx notebooks/plots
```
