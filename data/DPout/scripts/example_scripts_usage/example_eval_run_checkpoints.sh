python ../eval.py \
  --metric wer \
  --datasets atco_en_ruzyne \
  --datasets_basedir ../../data/atco \
  --models example_CHECKPOINTS \
  --checkpoints_eval \
  --same_processor \
  --output_file example_CHECKPOINTS/eval.txt \
  --batch_size 4 \
  --eval_description "" \
  --callsigns_name_in_ds long_callsigns \
  --transcription_name_in_ds full_ts \
  --prompt_name_in_ds prompt_fullts_AG_4B \
  --separate_ds \
  --ignore_case \
  --use_prompt \
  # --NOP_wer_for_AG_existing_only \
  # --eval_callsigns \

