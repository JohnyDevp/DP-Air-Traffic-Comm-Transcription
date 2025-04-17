#!/bin/bash
SERVER=merlin
ROOT=/mnt/scratch/tmp/xholan11
NUMBER_OF_EXPERIMENTS=3

# FULLTS
# DIR=models/planned-allds-fullts/30/dropout       # CHANGE THIS LINE
# EXPER_NAME_DIRS=do                              # CHANGE THIS LINE
# DST=./allds/30/dropout                           # CHANGE THIS LINE
# #======================================================
DIR=models/planned-vanmed-fullts/batchsize        # CHANGE THIS LINE
EXPER_NAME_DIRS=bs                                # CHANGE THIS LINE
DST=./vanmed/30/batchsize                           # CHANGE THIS LINE

for i in $(seq 1 $NUMBER_OF_EXPERIMENTS); do
    scp $SERVER:$ROOT/$DIR/$EXPER_NAME_DIRS${i}/eval.txt $DST/$EXPER_NAME_DIRS${i}_eval.txt
    scp $SERVER:$ROOT/$DIR/$EXPER_NAME_DIRS${i}/training_details.txt $DST/$EXPER_NAME_DIRS${i}_td.txt
    echo "DONE copying $SERVER:$ROOT/$DIR/$EXPER_NAME_DIRS${i}/{eval.txt, training_details.txt} to $DST"
done