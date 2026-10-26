# 环境配置（训练与重定向隔离）

## 1. 基础约束

- 系统：Ubuntu 22.04 / WSL2
- Python：3.11
- GPU：NVIDIA（示例机型 RTX 5060 Laptop GPU，约 8GB 显存）
- 不自动安装驱动，不修改系统网络或 WSL 配置

## 2. 训练环境（`.venv-holoboxing`）

```bash
python3.11 -m venv .venv-holoboxing
source .venv-holoboxing/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r "$HOLOMOTION_ROOT/holoboxing/training/requirements.txt"
python -m pip check
```

## 3. 动作重定向环境（`.venv-holoboxing-retarget`）

```bash
python3.11 -m venv .venv-holoboxing-retarget
.venv-holoboxing-retarget/bin/python -m pip install torch==2.9.1 --index-url https://download.pytorch.org/whl/cu128
.venv-holoboxing-retarget/bin/python -m pip install -r "$HOLOMOTION_ROOT/holoboxing/data/retarget-requirements.txt"
.venv-holoboxing-retarget/bin/python -m pip check
```

## 4. 路径变量建议

```bash
export HOLOMOTION_ROOT=/abs/path/to/HoloMotion
export TRAIN_PY=/abs/path/to/.venv-holoboxing/bin/python
export RETARGET_PY=/abs/path/to/.venv-holoboxing-retarget/bin/python
```

## 5. 资源目录约束

- `HOLOBOXING_ASSETS` 目录应含 `resources.json`
- `HOLOBOXING_LOWER` 目录应含 `config.yaml` 和 `exported/model_10500.onnx`
- 数据准备历史存在 OOM，默认建议 `workers=1`

## 6. 与官方接口差异说明

在 `f52dd81` 中：
- 存在：`python -m holoboxing assets/download`
- 存在：`python -m holoboxing.data download/retarget/prepare/verify`
- 不存在聚合命令：`python -m holoboxing.training ...`
- 应使用模块入口：
  - `python -m holoboxing.training.prepare_prior`
  - `python -m holoboxing.training.train_prior`
  - `python -m holoboxing.training.train_sl`
  - `python -m holoboxing.training.train_rl`
  - `python -m holoboxing.training.export`
