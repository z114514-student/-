# 弘锦操作步骤：反架专项微调

本仓库保存团队实验脚本。运行位置是已配置好的官方 HoloMotion 源码目录，不是本仓库。源码基线为 `f52dd81`。脚本不改官方代码，不自动上传模型或操作真实机器人。

## 1. 实验目的及边界

003镜像候选在短窗口底层跟踪中保持右手、右脚领先；第一版上层ONNX在这些观测上预测异侧领先。首轮实验冻结动作解码器和输入归一化，只微调上层网络。

- 训练：003镜像8.82～9.72秒、9.74～10.72秒（右端不包含），与原InterHuman训练样本各占每批一半。
- 验证：004镜像11.74～12.46秒，以及InterHuman验证集；不参与梯度更新。004仍需人工核对完整抱架质量。
- 使用原检查点的损失归一化，100次更新、学习率1e-5仅为初始实验设置。
- `right_lead`只代表自身heading坐标下的同侧前后关系，不代表防守姿态、碰撞、接触或闭环稳定性合格。
- 003用于训练，不能再作为独立测试集；004用于调参验证，最终测试需另设。

## 2. 获取团队脚本

在Linux/WSL中执行，选择尚不存在的目标目录：

```bash
mkdir -p ~/projects
git clone https://github.com/z114514-student/-.git ~/projects/hbot-team
```

如果已有此克隆，先检查未提交修改，再在对应发布分支更新；不要重复克隆或覆盖修改。

## 3. 准备独立交接资源

资源不包含在这次代码上传中。由算法负责人通过符合数据许可的渠道提供：

1. 完整的已校验预处理数据目录：保留prepared、robot_h5、sources.json、COMPLETE.json及内部相对路径。
2. 原始上层检查点：`results/boxing-sl/epoch_000020.pt`。
3. 动作表示检查点：`results/boxing-prior/best.pt`。
4. 第一版模型：`results/boxing-sl-v1.onnx`（主要供诊断脚本比较）。

微调本身不启动底层物理仿真，不需要重新下载1.65GB底层控制器；之后官方仿真评测仍需要官方资源。

不要把SMPL、受限数据、私人路径、凭据或完整训练数据上传到公开仓库。

## 4. 检查官方目录和环境

以下假定官方目录是 `~/projects/HoloMotion`，否则替换为实际位置：

```bash
cd ~/projects/HoloMotion
git rev-parse --short HEAD
source .venv-holoboxing/bin/activate
python --version
python -m pip check
ls results/boxing-sl/epoch_000020.pt
```

使用Python 3.11、PyTorch 2.9.1及项目训练依赖。不要在重定向环境中运行。本指南以已有环境为前提。

## 5. 写入本机数据位置

先找到解压后真正包含manifest.json的prepared目录；不要照抄发送方的用户名或目录。

```bash
mkdir -p logs
```

将下面示例路径替换为本机实际的绝对路径，再执行：

```bash
printf '%s\n' '/替换为实际目录/combined_xxx/prepared' > logs/prepared_data_path.txt
ls "$(cat logs/prepared_data_path.txt)/manifest.json"
```

最后一条必须成功。路径仅保存在本地，不提交GitHub。

## 6. 先做单步检查

在官方HoloMotion目录中运行团队仓库的脚本：

```bash
python ~/projects/hbot-team/scripts/finetune_guard.py \
  --smoke-test \
  --output "results/guard_smoke_$(date +%Y%m%d_%H%M%S)"
```

预期报告：`frozen_components_unchanged: true`、`latent_policy_changed: true`、`exported: false`。该模式只做一次更新以验证流程，不生成可交付模型。

如有数据缺失、哈希不符、分区不符，停止并修正来源；不要修改清单伪装成InterHuman。出现错误发完整命令和日志末尾。

## 7. 第一轮100步实验

```bash
GUARD_RUN="results/guard_run_$(date +%Y%m%d_%H%M%S)"
set -o pipefail
python -u ~/projects/hbot-team/scripts/finetune_guard.py \
  --steps 100 \
  --batch-size 32 \
  --lr 0.00001 \
  --device cpu \
  --output "$GUARD_RUN" \
  2>&1 | tee "logs/$(basename "$GUARD_RUN").log"
guard_exit=$?
printf '退出码：%s\n结果目录：%s\n' "$guard_exit" "$GUARD_RUN"
```

CPU模式不启动GPU仿真。只有退出码0才能视为程序完成；输出目录须是新目录，不覆盖已有结果。

## 8. 给算法负责人返回什么

- `config.json`：训练设置、实际数据来源和哈希。
- `report.json`：baseline与after，训练、验证指标及导出检查。
- `train.jsonl`：训练日志。
- 失败时发送终端日志，不只说“跑不了”。

优先比较guard_val的右侧领先计数、位置误差及original_val是否退化，不能只看guard训练样本。即便离线改善，也需独立仿真。

## 9. 模型交接与检查点格式

- `guard_candidate.onnx`：只有算法负责人审核报告后，交给西柚做官方校验、完整链路测时和回放对比。静态检查和CPU导出一致性不是平台验收。
- `guard_mixed.pt`：自定义格式`hbot_guard_mixed_v1`，包含混合训练来源、策略状态及优化器；不能直接传给官方train_rl、train_sl或export入口。
- 当前脚本不提供断点续训；每轮从原epoch_000020.pt开始。后续续训或RL衔接必须显式实现适配，保留真实来源。
- 不覆盖第一版模型。西柚比较抱架、滑脚、单脚跳动、摔倒及进入姿态的过程。

## 10. 其他诊断脚本

`check_guard_layers.py`检查原始第一版模型和动作解码器，不会自动分析新微调模型：

```bash
python ~/projects/hbot-team/scripts/check_guard_layers.py \
  --output "results/guard_layers_$(date +%Y%m%d_%H%M%S)"
```

`guard_diagnose.py`可筛查RoboStriker参考动作中的低速领先关系，以及列出回放字段：

```bash
python ~/projects/hbot-team/scripts/guard_diagnose.py scan \
  --source holoboxing/data/raw/RoboStriker \
  --output "results/guard_scan_$(date +%Y%m%d_%H%M%S)"
python ~/projects/hbot-team/scripts/guard_diagnose.py inspect /实际路径/replay.hdf5
```

## 11. GitHub网络与聊天网络分开

代理客户端可以只提供本地HTTP/Mixed端口，不启用系统代理或TUN。此时仅给需要访问GitHub的命令指定代理：

```bash
git -c http.proxy=http://127.0.0.1:33210 clone https://github.com/z114514-student/-.git ~/projects/hbot-team
```

上述端口是当前队长使用的示例，其他人应使用自己的实际端口。WSL通过localhost访问Windows代理还取决于镜像网络配置。不要把代理订阅、节点密码放入仓库。若客户端开启后仍影响聊天，需检查TUN、VPN及系统代理，不能认定单命令代理已经生效。
