# 数据到评测工作流

> 所有命令均基于官方提交 `f52dd81`；默认在本仓库执行，实际代码由 `HOLOMOTION_ROOT` 提供。

## 0) 环境与资源检查

```bash
scripts/check_env.sh --config configs/env.example.env
```

## 1) 数据下载与重定向

```bash
scripts/data_prepare.sh --config configs/data_prepare.example.env
```

包含：
- `holoboxing.data download robostriker`
- `holoboxing.data download interhuman`
- `holoboxing.data retarget`
- `holoboxing.data prepare`
- `holoboxing.data verify`

## 2) 动作表示训练（prior）

```bash
scripts/train_prior.sh --config configs/train_prior.example.env
```

包含：
- `holoboxing.training.prepare_prior`
- `holoboxing.training.train_prior`

## 3) 监督学习（SL）

```bash
scripts/train_sl.sh --config configs/train_sl.example.env
```

包含：
- `holoboxing.training.train_sl`

## 4) ONNX 导出

```bash
scripts/export_onnx.sh --config configs/export_eval.example.env
```

包含：
- `holoboxing.training.export`

## 5) 本地官方校验、测时、对战

```bash
scripts/eval_validate.sh validate --config configs/export_eval.example.env
scripts/eval_validate.sh benchmark --config configs/export_eval.example.env
scripts/eval_validate.sh match --config configs/export_eval.example.env --opponent /abs/path/to/opponent.onnx
```

使用官方入口：`$HOLOMOTION_ROOT/holoboxing/run.sh`

## 注意

- 校验失败时不要继续提交。
- 不自动上传赛事平台。
- 无真实 GPU 或模型时，可先跑脚本语法与单测，评测步骤标注“未执行”。
