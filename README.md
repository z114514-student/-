# H-Bot 机器人拳击协作仓库（团队版）

本仓库用于五人大学生团队协作管理 **HoloBoxing** 训练配置、实验记录、ONNX 导出与评测交付。

- 官方上游仓库：`HorizonRobotics/HoloMotion`
- 已核对上游提交：`f52dd81`
- 本仓库定位：**团队流程与配置仓库**（不重写官方仿真器、评分器、训练算法）

## 当前进度（真实状态）

- ✅ 已完成：数据准备、动作表示训练、20 轮监督学习
- ✅ 已导出：`boxing-sl-v1.onnx`
- ⚠️ 未确认：官方运行校验、时延测试、对战评测结果
- ⚠️ 因此该模型**不能**描述为“可晋级”或“已具备有效拳击能力”

## 团队分工

1. 算法负责人：策略训练、奖励设计、实验方案。
2. 嵌软负责人 A：环境、数据流水线和训练运行。
3. 嵌软负责人 B：模型导出、接口校验、推理性能和交付。
4. 动作分析负责人：坐标、轨迹、平衡和失败案例分析。
5. 评测负责人：实验记录、回放分析和结果统计。

## 官方源码与路径约定

- 本仓库通过 `HOLOMOTION_ROOT` 指向独立的官方源码目录。
- 不在本仓库内复制官方代码，不自动升级上游版本。
- 若官方接口与本文档不一致，请以 `f52dd81` 对应源码为准，并在文档记录差异。

## 快速开始（建议顺序）

1. 阅读 `docs/setup.md`，准备两个隔离虚拟环境：
   - `.venv-holoboxing`（训练/导出）
   - `.venv-holoboxing-retarget`（动作重定向）
2. 配置 `.env`（参考 `.env.example` 和 `configs/*.example.env`）。
3. 运行环境检查：`scripts/check_env.sh`
4. 数据与验证：`scripts/data_prepare.sh`
5. 动作表示训练：`scripts/train_prior.sh`
6. 监督学习：`scripts/train_sl.sh`
7. ONNX 导出：`scripts/export_onnx.sh`
8. 官方校验/测时/对战：`scripts/eval_validate.sh`
9. 交付前生成模型清单：
   `python3 tools/generate_model_manifest.py --onnx /path/model.onnx --source-checkpoint ... --training-stage ...`

> 说明：数据准备失败时，禁止自动进入训练。

## 常见问题入口

- 环境与依赖：`docs/setup.md`
- 全流程命令：`docs/workflow.md`
- 交付清单：`docs/handoff.md`
- 故障排查：`docs/troubleshooting.md`
- 团队阶段验收：`docs/team-plan.md`

## 许可与数据合规

- 不上传 SMPL 文件、InterHuman 原始数据或其他受许可限制资源。
- 不提交访问令牌、代理订阅、密码或个人路径。
- 不擅自变更官方代码或第三方数据许可证；文档保留来源与许可说明。
