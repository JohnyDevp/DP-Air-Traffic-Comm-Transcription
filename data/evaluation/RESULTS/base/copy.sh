#!/bin/bash
SERVER=merlin
ROOT=/mnt/scratch/tmp/xholan11

# FULLTS + SHORT
# "vanmed-allds-full","vanmed-allds-short", "vanmed-atcoen-short"
EVALUATED_MODEL_NAME="allds-atcoen-short"
DST_DIR="./shortts/allds-atcoen"

# for folder in "${EVALUATED_MODEL_NAMES[@]}"; do
scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/mypar/eval.txt                           $DST_DIR/mp_eval.txt
scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/mypar/training_details.txt               $DST_DIR/mp_td.txt
scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/verpar/eval.txt                          $DST_DIR/vp_eval.txt
scp $SERVER:$ROOT/models/$EVALUATED_MODEL_NAME/verpar/training_details.txt              $DST_DIR/vp_td.txt
# done

# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/eval.txt               ./$EVALUATED_MODEL_NAME/AG_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/training_details.txt   ./$EVALUATED_MODEL_NAME/AG_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/eval.txt             ./$EVALUATED_MODEL_NAME/AG4B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/training_details.txt ./$EVALUATED_MODEL_NAME/AG4B_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/eval.txt               ./$EVALUATED_MODEL_NAME/5B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/training_details.txt   ./$EVALUATED_MODEL_NAME/5B_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/eval.txt              ./$EVALUATED_MODEL_NAME/50B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/training_details.txt  ./$EVALUATED_MODEL_NAME/50B_td.txt
