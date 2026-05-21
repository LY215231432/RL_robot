# IsaacLab Training Notes

这个目录现在只放 IsaacLab 训练相关的说明资料，不再展开整理 MuJoCo 侧工程。

建议按下面顺序看：

1. [PROJECT_FILE_GUIDE.md](/E:/IsaacLab-main/mywork/PROJECT_FILE_GUIDE.md)
   先看这份总览，知道训练链路涉及哪些源码、配置和结果文件。

2. [xarm7_isaaclab_group_meeting.md](/E:/IsaacLab-main/mywork/isaaclab_training/xarm7_isaaclab_group_meeting.md)
   这份更适合组会讲解，按“机器人接入 -> 任务定义 -> reward -> PPO -> 结果”的顺序组织。

3. [isaaclab_h5py_check.txt](/E:/IsaacLab-main/mywork/isaaclab_training/isaaclab_h5py_check.txt)
   这是运行 `play.py` 时排查 `h5py` 导入问题的辅助记录。

补充：

- `xarm7_isaaclab_group_meeting.html` 是较早导出的 HTML 版本。
- 目前以 Markdown 文档为准，后续优先维护 `.md`。

真正训练时最关键的源码和结果位置仍然在：

- `source/isaaclab_assets/isaaclab_assets/robots/xarm7.py`
- `source/isaaclab_tasks/isaaclab_tasks/manager_based/manipulation/lift/config/xarm7/`
- `logs/rsl_rl/xarm7_lift/2026-04-08_12-42-49/`

VS Code 里现在也可以直接从这些入口启动：

- `.vscode/launch.json` 里的 `Python: Train xArm7 Lift`
- `.vscode/launch.json` 里的 `Python: Play xArm7 Lift`
- `.vscode/tasks.json` 里的 `xarm7_train_headless`
- `.vscode/tasks.json` 里的 `xarm7_record_video`
