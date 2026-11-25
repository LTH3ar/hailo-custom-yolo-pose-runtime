#!/bin/bash
until grep -q QUEUE_DONE ~/hcpose_paper/h07/logs/queue.log 2>/dev/null; do sleep 60; done
~/hcpose_paper/h07/tools/cpu_phase.sh > ~/hcpose_paper/h07/logs/cpu.out 2>&1
