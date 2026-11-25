#!/usr/bin/env bash
cd ~/hailo-custom-yolo-pose-runtime
for g in current offset yaw; do
  nice .venv/bin/python /home/raven/hcpose_paper/tools/sweep_rules_h02.py --grid $g --jobs 3 --work /tmp/h02_sweep --out /home/raven/hcpose_paper/sweep/sweep_$g.md > /home/raven/hcpose_paper/sweep/sweep_$g.log 2>&1
  echo "$g rc=$?"
done
echo SWEEP_DONE
