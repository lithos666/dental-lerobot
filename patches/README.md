# LeRobot 上游兼容层补丁（patches/）

本项目的 `src/lerobot/` 基于上游 [huggingface/lerobot](https://github.com/huggingface/lerobot)
（基线提交 `8b4dcb14`，即 `origin/mypy_support` / `origin/pre-commit-ci-update-config` 所在线）。
为牙科机器人项目所做的全部适配修改已提取为本目录下的补丁文件，
以便未来同步上游新版本时可以用 `git apply` 一键重放，降低合并冲突成本。

## 补丁清单

| 补丁 | 涉及文件 | 作用 |
|---|---|---|
| `0001-opencv-flip-horizontal.patch` | `cameras/opencv/camera_opencv.py`、`cameras/opencv/configuration_opencv.py` | 新增 `flip_horizontal` 选项：腕部相机水平翻转安装，画面需镜像还原 |
| `0002-motors-bus-retry.patch` | `motors/feetech/feetech.py`、`motors/motors_bus.py` | USB-串口适配器（CH343）偶发丢包时自动重试 |
| `0003-so101-wraparound-guard.patch` | `robots/so101_follower/so101_follower.py` | STS3215 位置回绕（wrap-around）防护：检测异常跳变并回退到上次有效位置，防止安全钳位把舵机推向错误方向 |
| `0004-configs-windows-tempfile.patch` | `configs/policies.py` | Windows 平台临时文件兼容修复 |
| `0005-ruff-sim-lint.patch` | `datasets/lerobot_dataset.py`、`processor/observation_processor.py` 等 | 启用 Ruff SIM 规则后的纯 lint 修复（无行为变化），上游同步时若冲突可直接丢弃并重新 `ruff check --fix` |

> 0001~0004 是功能性适配，必须保留；0005 只是代码风格，可按需重做。

## 同步上游后重新应用

```powershell
# 在本仓库根目录，假设刚把 src/lerobot/ 更新到了新的上游版本
$patches = (Get-ChildItem patches\0*.patch).FullName
git apply --check $patches   # 先干跑检查是否有冲突
git apply $patches           # 无冲突则直接应用

# 或直接运行封装脚本：
powershell -File patches\apply_patches.ps1
```

出现冲突时（上游改动了同一区域），`git apply` 会指出具体文件，
可加 `--3way` 走三方合并，或手工对照补丁内容迁移修改，
然后更新对应补丁文件：

```powershell
# 改完代码后，针对新基线重新生成某个补丁
git diff --output=patches/0003-so101-wraparound-guard.patch <新基线> HEAD -- src/lerobot/robots/so101_follower/so101_follower.py
```

## 校验方式

补丁在基线 `8b4dcb14` 的干净检出上通过 `git apply --check` 验证。
若仓库保留了基线分支，可自查：

```powershell
git worktree add --detach $env:TEMP\patch_check origin/mypy_support
Set-Location $env:TEMP\patch_check
git apply --check (Get-ChildItem <本仓库>\patches\0*.patch).FullName
```
