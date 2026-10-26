# 常见故障排查

## 1. 代理连接问题

- 检查是否设置了 `HTTP_PROXY/HTTPS_PROXY/NO_PROXY`。
- 若公司代理策略变化，先用 `curl` 验证外网连通性。
- 配置文件中不要提交真实代理地址或凭据。

## 2. 文件路径错误

- 优先使用绝对路径。
- 检查 `HOLOMOTION_ROOT` 是否指向包含 `holoboxing/run.sh` 的目录。
- 检查 `TRAIN_PY` 与 `RETARGET_PY` 是否指向不同虚拟环境。

## 3. Chumpy 构建失败

- 在重定向环境重新安装：
  `RETARGET_PY -m pip install -r holoboxing/data/retarget-requirements.txt`
- 执行 `pip check` 查依赖冲突。
- 必要时清理 pip 缓存后重装。

## 4. 退出码 137 / OOM

- 137 通常表示被系统杀死（内存不足）。
- 数据准备默认 `workers=1`，减少并发。
- 降低批大小、缩小数据子集做冒烟验证。
- 不将某一台机器的 WSL 内存配置当成团队硬要求。

## 5. 数据清单缺失

- `HOLOBOXING_ASSETS` 应含 `resources.json`。
- 运行：`python -m holoboxing.data verify --output ... --assets "$HOLOBOXING_ASSETS"`
- 若缺文件，回到下载/准备步骤重跑并检查日志。
