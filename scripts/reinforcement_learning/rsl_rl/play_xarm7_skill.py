# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Thin launcher that plays an xArm7 skill by skill name."""

from __future__ import annotations

import argparse
import importlib.util
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
REGISTRY_PATH = (
    REPO_ROOT
    / "source"
    / "isaaclab_tasks"
    / "isaaclab_tasks"
    / "manager_based"
    / "manipulation"
    / "xarm7_skill_lib"
    / "registry.py"
)


def _load_skill_registry():
    module_name = "xarm7_skill_registry"
    module_spec = importlib.util.spec_from_file_location(module_name, REGISTRY_PATH)
    if module_spec is None or module_spec.loader is None:
        raise RuntimeError(f"Unable to load skill registry from {REGISTRY_PATH}.")

    module = importlib.util.module_from_spec(module_spec)
    module_spec.loader.exec_module(module)
    return module.SKILL_REGISTRY


SKILL_REGISTRY = _load_skill_registry()


def _print_skills():
    print("Available xArm7 skills:")
    for skill_name, spec in sorted(SKILL_REGISTRY.items()):
        suffix = f" | checkpoint: {spec.default_checkpoint}" if spec.default_checkpoint else ""
        print(f"  - {skill_name}: {spec.play_task}{suffix}")
        print(f"    {spec.description}")


def main():
    parser = argparse.ArgumentParser(description="Play an xArm7 skill through the skill registry.")
    parser.add_argument("--skill", type=str, help="Skill name registered in xarm7_skill_lib.registry.")
    parser.add_argument("--list", action="store_true", help="List available skills and exit.")
    args, passthrough = parser.parse_known_args()

    if args.list:
        _print_skills()
        return 0

    if not args.skill:
        parser.error("--skill is required unless --list is used.")

    if args.skill not in SKILL_REGISTRY:
        print(f"Unknown skill: {args.skill}\n")
        _print_skills()
        return 1

    spec = SKILL_REGISTRY[args.skill]
    target_script = Path(__file__).with_name("play.py")

    command = [sys.executable, str(target_script), "--task", spec.play_task]
    if spec.default_checkpoint and "--checkpoint" not in passthrough:
        checkpoint = Path(spec.default_checkpoint)
        if not checkpoint.is_absolute():
            checkpoint = REPO_ROOT / checkpoint
        command.extend(["--checkpoint", str(checkpoint)])
    command.extend(passthrough)

    print(f"[INFO] Playing skill '{args.skill}' using task '{spec.play_task}'.")
    print("[INFO] Command:", " ".join(command))
    result = subprocess.run(command, cwd=REPO_ROOT, check=False)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
