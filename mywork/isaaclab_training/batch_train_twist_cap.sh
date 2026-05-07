#!/usr/bin/env bash
set -euo pipefail

# Batch training launcher for xArm7 twist-cap reward/action ablations.
#
# Usage on the Linux server:
#   cd /home/exp1/fyd/robot/ly/IsaacLab-main
#   chmod +x mywork/isaaclab_training/batch_train_twist_cap.sh
#   NUM_ENVS=256 MAX_ITERATIONS=4000 ./mywork/isaaclab_training/batch_train_twist_cap.sh
#
# Useful controls:
#   RUNS="soft_gate pose_twist" NUM_ENVS=128 MAX_ITERATIONS=1000 ./mywork/isaaclab_training/batch_train_twist_cap.sh
#   DRY_RUN=1 ./mywork/isaaclab_training/batch_train_twist_cap.sh

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "${REPO_ROOT}"

NUM_ENVS="${NUM_ENVS:-256}"
MAX_ITERATIONS="${MAX_ITERATIONS:-4000}"
SEED="${SEED:-42}"
DEVICE="${DEVICE:-cuda:0}"
HEADLESS="${HEADLESS:-1}"
DRY_RUN="${DRY_RUN:-0}"
RUNS="${RUNS:-soft_gate contact_easy long_horizon pose_twist}"

LOG_DIR="${REPO_ROOT}/logs/rsl_rl/xarm7_twist_cap_batch"
mkdir -p "${LOG_DIR}"

base_cmd=(
    "./isaaclab.sh"
    "-p"
    "scripts/reinforcement_learning/rsl_rl/train_xarm7_skill.py"
    "--skill"
    "twist_cap"
    "--num_envs"
    "${NUM_ENVS}"
    "--max_iterations"
    "${MAX_ITERATIONS}"
    "--seed"
    "${SEED}"
    "--device"
    "${DEVICE}"
)

if [[ "${HEADLESS}" == "1" ]]; then
    base_cmd+=("--headless")
fi

run_experiment() {
    local name="$1"
    shift
    local timestamp
    timestamp="$(date +%Y%m%d_%H%M%S)"
    local log_file="${LOG_DIR}/${timestamp}_${name}.log"
    local cmd=("${base_cmd[@]}" "agent.run_name=${name}" "$@")

    echo
    echo "============================================================"
    echo "[INFO] Starting twist_cap experiment: ${name}"
    echo "[INFO] Log file: ${log_file}"
    printf '[INFO] Command:'
    printf ' %q' "${cmd[@]}"
    echo
    echo "============================================================"

    if [[ "${DRY_RUN}" == "1" ]]; then
        return 0
    fi

    "${cmd[@]}" 2>&1 | tee "${log_file}"
}

for run_name in ${RUNS}; do
    case "${run_name}" in
        soft_gate)
            # Goal: make rotation rewards visible before the grasp score is perfect.
            run_experiment "${run_name}" \
                "env.episode_length_s=10.0" \
                "env.rewards.grasp_cap.params.descend_gate_threshold=0.30" \
                "env.rewards.twist_tcp.weight=1.5" \
                "env.rewards.twist_tcp.params.grasp_gate_threshold=0.30" \
                "env.rewards.rotate_cap.weight=4.0" \
                "env.rewards.rotate_cap.params.grasp_gate_threshold=0.30" \
                "env.rewards.rotate_cap_velocity.weight=0.5" \
                "env.rewards.rotate_cap_velocity.params.grasp_gate_threshold=0.30" \
                "env.curriculum.twist_tcp.params.num_steps=8000" \
                "env.curriculum.rotate_cap.params.num_steps=8000" \
                "env.curriculum.rotate_cap_velocity.params.num_steps=8000"
            ;;

        contact_easy)
            # Goal: reduce sparse-contact blocking by making bilateral contact score easier to express.
            run_experiment "${run_name}" \
                "env.episode_length_s=10.0" \
                "env.rewards.grasp_cap.weight=14.0" \
                "env.rewards.grasp_cap.params.contact_force_scale=2.0" \
                "env.rewards.grasp_cap.params.descend_gate_threshold=0.25" \
                "env.rewards.twist_tcp.weight=1.0" \
                "env.rewards.twist_tcp.params.contact_force_scale=2.0" \
                "env.rewards.twist_tcp.params.grasp_gate_threshold=0.25" \
                "env.rewards.rotate_cap.weight=3.0" \
                "env.rewards.rotate_cap.params.contact_force_scale=2.0" \
                "env.rewards.rotate_cap.params.grasp_gate_threshold=0.25" \
                "env.rewards.rotate_cap_velocity.weight=0.5" \
                "env.rewards.rotate_cap_velocity.params.contact_force_scale=2.0" \
                "env.rewards.rotate_cap_velocity.params.grasp_gate_threshold=0.25"
            ;;

        long_horizon)
            # Goal: give the full approach-descend-grasp-rotate chain more time.
            run_experiment "${run_name}" \
                "env.episode_length_s=12.0" \
                "agent.num_steps_per_env=48" \
                "env.rewards.find_cap_above.weight=7.0" \
                "env.rewards.descend_to_grasp.weight=8.0" \
                "env.rewards.grasp_cap.weight=12.0" \
                "env.rewards.rotate_cap.weight=2.5" \
                "env.rewards.rotate_cap.params.grasp_gate_threshold=0.35" \
                "env.rewards.rotate_cap_velocity.weight=0.4" \
                "env.rewards.rotate_cap_velocity.params.grasp_gate_threshold=0.35"
            ;;

        pose_twist)
            # Goal: test whether 6D relative pose IK helps the policy learn TCP yaw/twist.
            # This changes action dimensionality, so train this run from scratch.
            run_experiment "${run_name}" \
                "env.episode_length_s=10.0" \
                "env.actions.arm_action.controller.command_type=pose" \
                "env.actions.arm_action.scale=[0.04,0.04,0.03,0.0,0.0,0.25]" \
                "env.rewards.grasp_cap.params.descend_gate_threshold=0.30" \
                "env.rewards.twist_tcp.weight=3.0" \
                "env.rewards.twist_tcp.params.grasp_gate_threshold=0.30" \
                "env.rewards.rotate_cap.weight=5.0" \
                "env.rewards.rotate_cap.params.grasp_gate_threshold=0.30" \
                "env.rewards.rotate_cap_velocity.weight=0.8" \
                "env.rewards.rotate_cap_velocity.params.grasp_gate_threshold=0.30" \
                "env.curriculum.twist_tcp.params.num_steps=6000" \
                "env.curriculum.rotate_cap.params.num_steps=6000" \
                "env.curriculum.rotate_cap_velocity.params.num_steps=6000"
            ;;

        *)
            echo "[ERROR] Unknown run '${run_name}'. Valid runs: soft_gate contact_easy long_horizon pose_twist" >&2
            exit 1
            ;;
    esac
done

echo
echo "[INFO] Batch training completed."
echo "[INFO] IsaacLab run directories: logs/rsl_rl/xarm7_twist_cap/"
echo "[INFO] Batch launcher logs: ${LOG_DIR}"
