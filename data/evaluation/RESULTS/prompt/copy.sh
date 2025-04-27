#!/bin/bash
SERVER=merlin
ROOT=/mnt/scratch/tmp/xholan11

# FULLTS + SHORT
# "vanmed-allds-full","vanmed-allds-short", "vanmed-atcoen-short"
EVALUATED_MODEL_NAME="alldsatcoen-short"
DST_DIR="./shortts/alldsatcoen-short"

list=("40B" "5B" "50CZB" "AG40B" "AG4B" "AG" "AG50CZB")
for folder in "${list[@]}"; do
    scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/$folder/eval.txt                           $DST_DIR/$folder\_eval.txt
    scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/$folder/training_details.txt               $DST_DIR/$folder\_td.txt
done

# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/eval.txt               ./$EVALUATED_MODEL_NAME/AG_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/training_details.txt   ./$EVALUATED_MODEL_NAME/AG_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/eval.txt             ./$EVALUATED_MODEL_NAME/AG4B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/training_details.txt ./$EVALUATED_MODEL_NAME/AG4B_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/eval.txt               ./$EVALUATED_MODEL_NAME/5B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/training_details.txt   ./$EVALUATED_MODEL_NAME/5B_td.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/eval.txt              ./$EVALUATED_MODEL_NAME/50B_eval.txt
# scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/training_details.txt  ./$EVALUATED_MODEL_NAME/50B_td.txt
