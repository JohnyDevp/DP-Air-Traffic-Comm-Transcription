#!/bin/bash
echo "Running delay script, waiting for 5 hours..."
sleep 16200
echo "Starting the run.sh in te-allds-short folder on merlin server..."
ssh merlin "cd /mnt/scratch/tmp/xholan11/sge_scripts/PROMPT/te-allds-short && ./submit.sh"
echo "Train jobs submited."
echo "Waiting for 2 hours for evaluation..."
sleep 7200
echo "Starting the run.sh in te-allds-short folder on merlin server..."
ssh merlin "cd /mnt/scratch/tmp/xholan11/sge_scripts/PROMPT/te-allds-short && ./run.sh"
echo "Run script completed."

