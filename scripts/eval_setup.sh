python eval.py \
  --metric wer \
  --datasets atco_en_ruzyne \
  --datasets_basedir ../data/atco/PROMPT/newshuffled_prompt \
  --models openai/whisper-tiny \
  --same_processor \
  --output_file ./test-ENCODER/eval.txt \
  --batch_size 4 \
  --eval_description "" \
  --callsigns_name_in_ds long_callsigns \
  --transcription_name_in_ds full_ts \
  --prompt_name_in_ds prompt_fullts_AG_4B \
  --separate_ds \
  --ignore_case \
  --use_prompt \
  # --eval_callsigns \
