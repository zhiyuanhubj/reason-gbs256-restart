#!/usr/bin/env bash
# Mirror last-rank logging.jsonl to wandb without restarting training.
set -euo pipefail
if [[ "$(hostname)" == ip-10-1-115-182 ]]; then
    echo "[ERR] refusing on login $(hostname)" >&2
    exit 2
fi
while IFS= read -r var; do
    unset "$var"
done < <(env | awk -F= '/^SLURM_/ {print $1}')
unset WANDB_RUN_ID WANDB_RESUME WANDB_RUN_GROUP WANDB_SWEEP_ID WANDB_API_KEY_HAOYU

export WANDB_API_KEY="$(tr -d ' \n' < /fsx/home/zhiyuan/.config/wandb/opensima_mirror.key)"
export WANDB_ENTITY="${WANDB_ENTITY:-zhiyuan-hu-bj-nus}"
export WANDB_PROJECT="${WANDB_PROJECT:-opensima}"
export WANDB_MODE=online
export WANDB_SILENT=true
test -n "$WANDB_API_KEY"

RUN_NAME="${RUN_NAME:-general_reason_0916_4n_20260921T133220Z}"
JSONL="${JSONL:-/opt/dlami/nvme/zhiyuan-reasoning-jsonl/checkpoints/${RUN_NAME}/logging.jsonl}"
SIDECAR_DIR="${SIDECAR_DIR:-/fsx/home/zhiyuan/logs/wandb_sidecar_${RUN_NAME}_${WANDB_ENTITY}}"
mkdir -p "$SIDECAR_DIR"
export JSONL RUN_NAME SIDECAR_DIR WANDB_ENTITY WANDB_PROJECT

exec /fsx/home/zhiyuan/miniconda3/envs/megatron-sft/bin/python -u - <<'PY'
import json, os, time
from pathlib import Path
import wandb

jsonl = Path(os.environ['JSONL'])
run_name = os.environ['RUN_NAME']
sidecar_dir = Path(os.environ['SIDECAR_DIR'])
sidecar_dir.mkdir(parents=True, exist_ok=True)
print(f'[sidecar] waiting for {jsonl} host={os.uname().nodename}', flush=True)
while not jsonl.exists():
    time.sleep(5)
print(f'[sidecar] found jsonl size={jsonl.stat().st_size}', flush=True)

entity = os.environ.get('WANDB_ENTITY', 'zhiyuan-hu-bj-nus')
project = os.environ.get('WANDB_PROJECT', 'opensima')
run = wandb.init(
    dir=str(sidecar_dir),
    project=project,
    entity=entity,
    name=run_name,
    mode='online',
    resume='never',
    settings=wandb.Settings(quiet=True, save_code=False, disable_code=True, disable_git=True),
)
print(f'[sidecar] wandb {run.entity}/{run.project} id={run.id} url={run.url}', flush=True)
(sidecar_dir / 'run_url.txt').write_text(run.url + '\n')
(sidecar_dir / 'run_id.txt').write_text(run.id + '\n')

offset = 0
last_step = -1
partial = ''
while True:
    if not jsonl.exists():
        time.sleep(2)
        continue
    with jsonl.open('r', encoding='utf-8') as f:
        f.seek(offset)
        chunk = f.read()
        offset = f.tell()
    if not chunk:
        time.sleep(3)
        continue
    data = partial + chunk
    lines = data.split('\n')
    partial = lines[-1]
    for line in lines[:-1]:
        line = line.strip()
        if not line:
            continue
        rec = json.loads(line)
        step = rec.get('iteration')
        if isinstance(step, str) and '/' in step:
            step = int(step.split('/')[0])
        elif not isinstance(step, int):
            step = last_step + 1
        logs = {}
        for k, v in rec.items():
            if isinstance(v, bool) or not isinstance(v, (int, float)):
                continue
            logs[f'train/{k}'] = v
        if not logs:
            continue
        wandb.log(logs, step=step)
        last_step = step
        print(f'[sidecar] logged step={step} keys={len(logs)}', flush=True)
PY
