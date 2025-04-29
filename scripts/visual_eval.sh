python visual_eval_sentences.py \
    --model openai/whisper-tiny \
    --dataset atco_en_ruzyne \
    --dataset_root ./data/ \
    --output_file visual_eval.txt \
    --use_prompt \
    --prompt_name prompt_fullts_AG \
    --transcription_name full_ts 