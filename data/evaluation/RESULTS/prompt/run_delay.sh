#!/bin/bash
echo "Running delay script, waiting for 3 hours..."
sleep 10800
echo "Starting the run.sh in te-alldsatcoen-full folder on merlin server..."
ssh merlin "cd /mnt/scratch/tmp/xholan11/sge_scripts/PROMPT/te-alldsatcoen-full && ./run.sh"
echo "Run script completed."