#!/bin/bash
# 8 节点 × 每节点 8 GPU：共 8 个 Slurm 任务（每节点 1 个 rank），进程内再开 GPU 池（动态池 / slurm-shard）。
#
# 提交（建议在仓库根 khl_workspace 下执行，便于 #SBATCH 日志落到 logs/）::
#   cd /path/to/khl_workspace
#   sbatch RoboTwin_khl/wm_scripts/collect_slurm_dynamic_pool_8n64g.sbatch
#
# 可选环境变量（与 collect_slurm_multinode.sbatch 一致）::
#   COLLECT_ROOT          默认 robotwin_generated
#   COLLECT_TASK_CONFIG   默认 demo_clean（无 .yml 后缀也可，由采集脚本解析）
#
# 日志::
#   logs/collect-dyn-8n64g-<jobid>.out / .err
#
# 依赖::
#   worldarena_robotwin_labeler/run_collect_slurm_shard.py  （若缺失需从备份恢复）

#SBATCH -J robotwin-dyn-8n64g
#SBATCH -p gpu
#SBATCH -N 8
#SBATCH --ntasks=8
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=16
#SBATCH --time=48:00:00
# 相对路径相对「执行 sbatch 时的当前目录」；请在 khl_workspace 根下提交本脚本。
#SBATCH -o logs/collect-dyn-8n64g-%j.out
#SBATCH -e logs/collect-dyn-8n64g-%j.err

set -euo pipefail

# Slurm 常把 batch 脚本拷到 /var/spool/slurmd/...，此时 BASH_SOURCE 不在 NFS 上；
# 用 SLURM_SUBMIT_DIR（sbatch 时的 cwd）作为仓库根，脚本路径用固定相对布局。
if [ -n "${SLURM_SUBMIT_DIR:-}" ]; then
  WORKSPACE=$(cd "$SLURM_SUBMIT_DIR" && pwd)
  SCRIPT_DIR="$WORKSPACE/RoboTwin_khl/wm_scripts"
else
  SCRIPT_DIR=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
  WORKSPACE=$(cd "$SCRIPT_DIR/../.." && pwd)
fi
cd "$WORKSPACE"
mkdir -p "$WORKSPACE/logs"

export CUDA_DEVICE_ORDER=PCI_BUS_ID

source /disk/worldmodel/miniconda3/etc/profile.d/conda.sh
conda activate RoboTwin

CPUS=$((SLURM_CPUS_PER_TASK * SLURM_NTASKS))
echo "JobID=$SLURM_JOB_ID NODES=$SLURM_JOB_NUM_NODES NTASKS=$SLURM_NTASKS CPUS=$CPUS"
echo "NODELIST=$SLURM_JOB_NODELIST"
echo "WORKSPACE=$WORKSPACE"
# 部分集群上 batch 里的 srun 不会自动用满 #SBATCH 的 --ntasks，只起 1 个 task → 只有一台机跑满 GPU。
# 显式传入与作业一致的 task 数（覆盖提交行时请保持 -n 与节点规划一致，例如 8 机动态池用 -n 8 --ntasks-per-node=1）。
echo "SLURM_JOB_NUM_NODES=$SLURM_JOB_NUM_NODES SLURM_NTASKS=$SLURM_NTASKS"
srun --ntasks="${SLURM_NTASKS}" --ntasks-per-node=1 \
  python3 "$SCRIPT_DIR/worldarena_robotwin_labeler/run_collect_slurm_shard.py" \
  --root "${COLLECT_ROOT:-robotwin_generated}" \
  --task-config "${COLLECT_TASK_CONFIG:-demo_clean}" \
  --slurm-shard
