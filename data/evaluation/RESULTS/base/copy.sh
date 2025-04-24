#!/bin/bash
SERVER=merlin
ROOT=/mnt/scratch/tmp/xholan11

# FULLTS + SHORT
# "vanmed-allds-full","vanmed-allds-short", "vanmed-atcoen-short"
EVALUATED_MODEL_NAMES=("vanmed-atcoen-full")
for folder in "${list[@]}"; do
    scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/mypar/eval.txt                           ./fullts/$EVALUATED_MODEL_NAME/mp_eval.txt
    scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/mypar/training_details.txt               ./fullts/$EVALUATED_MODEL_NAME/mp_td.txt
    scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/verpar/eval.txt                           ./fullts/$EVALUATED_MODEL_NAME/vp_eval.txt
    scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/verpar/training_details.txt               ./fullts/$EVALUATED_MODEL_NAME/vp_td.txt
done

# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/eval.txt               ./$EVALUATED_MODEL_NAME/AG_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/training_details.txt   ./$EVALUATED_MODEL_NAME/AG_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/eval.txt             ./$EVALUATED_MODEL_NAME/AG4B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/training_details.txt ./$EVALUATED_MODEL_NAME/AG4B_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/eval.txt               ./$EVALUATED_MODEL_NAME/5B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/training_details.txt   ./$EVALUATED_MODEL_NAME/5B_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/eval.txt              ./$EVALUATED_MODEL_NAME/50B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/training_details.txt  ./$EVALUATED_MODEL_NAME/50B_td.txt
