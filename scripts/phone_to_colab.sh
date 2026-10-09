#!/usr/bin/env bash
# Recorded phone session -> retargeted episodes -> LeRobot dataset -> Drive (for colab/dcad.py runs).
# Run the interactive steps first (RECORDING.md section 8): scripts/calib_colors.py, and click_layout.py if needed.
#
#   scripts/phone_to_colab.sh data/raw_phone/<session> [dataset name, default phone_v2]
#
# Then: python colab/dcad.py run V1_phone_naive V2_phone_ambient_t03
set -euo pipefail
cd "$(dirname "$0")/.."
SESSION=${1:?usage: $0 data/raw_phone/<session> [name]}
NAME=${2:-phone_v2}
PY=${PY:-python}
S=$(basename "$SESSION")

$PY scripts/process_phone.py "$SESSION"                    # -> data/human/$S/*.npz, replay report in outputs/m2
$PY vla/export_lerobot.py --src "data/human/$S" --sources human --tasks all --per_task 1000 --name "$NAME"
$PY - "$NAME" <<'EOF'
import json, sys
e = json.load(open(f"data/lerobot/{sys.argv[1]}/episodes.json"))
eps = e["episodes"]
ok = sum(ep["success"] for ep in eps)
print(f"{sys.argv[1]}: {len(eps)} episodes, replay success {ok}/{len(eps)}, env {e.get('env_cfg')}")
by = {}
for ep in eps:
    by.setdefault(ep["task"], []).append(ep["success"])
for t, v in sorted(by.items()):
    print(f"  {t:14s} {len(v)} clips, {sum(v)} replay successes")
EOF
python3 colab/dcad.py push-data "data/lerobot/$NAME"
