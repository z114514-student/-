# 算法负责人 → 评测负责人交付清单

每次交付至少包含：

1. 模型文件：`*.onnx`（< 50,000,000 字节）
2. 模型清单 JSON（由 `tools/generate_model_manifest.py` 生成）
3. 训练来源信息：
   - 源 checkpoint 路径
   - 训练阶段（如 `sl-epoch20`、`rl-update128`）
4. 运行配置快照：对应 `configs/*.env`（去敏）
5. 实验记录：`experiments/` 下对应条目
6. 官方运行验证结果（独立文件）：
   - validate 输出
   - benchmark 输出
   - match 输出

## 交付判定原则

- 静态检查通过（接口/大小/哈希）≠ 官方验收通过。
- 只有 `run.sh` 验证和评测结果可作为官方运行依据。
