# 标定输出缓存指引（calibration_cache/）

`dental_robot/` 下的 4 个标定产物记录了**本机硬件专属参数**，被 `.gitignore`
排除（换机器必须重新标定），一旦误删就要重跑整套标定流程（约 30 分钟）：

| 文件 | 由哪个脚本生成 | 作用 |
|---|---|---|
| `scene_camera_intrinsics.npz` | `calibrate_scene_camera.py` | 环境相机内参/畸变（ArUco 定位的前提） |
| `pan_mapping.json` | `calibrate_pan_mapping.py` | 方位角 -> 底座 pan 角的映射 |
| `scene_geometry.json` | `calibrate_scene_geometry.py` | 基座 marker 位姿 + 标准牙位 |
| `start_pose.json` | `calibrate_start_pose.py` | 机械臂规范起始姿态（ID 2-6） |

## 建议做法：本机备份一份快照

把标定结果复制进本目录（已在 `.gitignore` 中排除，仅 README 入库）：

```powershell
# 备份（标定全部完成后执行一次）
New-Item -ItemType Directory -Force calibration_cache | Out-Null
Copy-Item dental_robot\scene_camera_intrinsics.npz, dental_robot\pan_mapping.json, `
    dental_robot\scene_geometry.json, dental_robot\start_pose.json calibration_cache\
```

```powershell
# 恢复（文件丢失/误删时，免去重新标定）
Copy-Item calibration_cache\scene_camera_intrinsics.npz, calibration_cache\pan_mapping.json, `
    calibration_cache\scene_geometry.json, calibration_cache\start_pose.json dental_robot\
```

## 注意事项

- 缓存文件**只在产生它的那台机器/那套硬件上有效**：换相机、换机械臂、
  挪动相机安装位置后，必须重新标定并覆盖缓存。
- 备份前确认 4 个文件都是最新一轮标定的产物（查看文件修改时间）。
- 相机索引（`config.py` 的 `SCENE_CAM_INDEX` / `HANDEYE_CAM_INDEX`）
  不属于标定产物，换了 USB 口后要按操作手册第八章的方法重新确认。
