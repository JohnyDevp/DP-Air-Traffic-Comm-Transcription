#!/bin/bash
SERVER=merlin
ROOT=/mnt/scratch/tmp/xholan11

EVALUATED_MODEL_NAME=allds_lr1

scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/eval.txt               ./$EVALUATED_MODEL_NAME/AG_eval.txt
scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG/training_details.txt   ./$EVALUATED_MODEL_NAME/AG_td.txt
scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/eval.txt             ./$EVALUATED_MODEL_NAME/AG4B_eval.txt
scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/AG4B/training_details.txt ./$EVALUATED_MODEL_NAME/AG4B_td.txt
scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/eval.txt               ./$EVALUATED_MODEL_NAME/5B_eval.txt
scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/5B/training_details.txt   ./$EVALUATED_MODEL_NAME/5B_td.txt
scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/eval.txt              ./$EVALUATED_MODEL_NAME/50B_eval.txt
scp $SERVER:$ROOT/models/PROMPT/$EVALUATED_MODEL_NAME/50B/training_details.txt  ./$EVALUATED_MODEL_NAME/50B_td.txt
