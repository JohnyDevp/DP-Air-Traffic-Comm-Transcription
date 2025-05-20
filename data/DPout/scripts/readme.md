## AUTOR
Jan Holáň, xholan11
xholan11@stud.fit.vutbr.cz

## POPIS
V této složce jsou obsažené skripty pro trénování, testování a vizualizaci přepisů jednotlivými modely.

Očekává se spuštění v prostředí python3.12 s balíčky definovanými v adresáři *env*

## eval.py
Skript pro vyhodnocování úspěšnosti jednotlivých modelů. Jediná implementovaná metrika je wer.

Parametry, které přijímá lze získat spuštěním `$ python eval.py -h`
Názvy datasetů pro testování mohou být následující:
- atco_en_ruzyne (en_ruzyne_test_ds)
- atco_en_stefanik (en_stefanik_test_ds)
- atco_en_zurich (en_zurich_test_ds)
- atco_fr (fr_test_ds)
- atco_other_lang (other_lang_test_ds)
- hiwire_fr (hwir_fr_test_ds)
- hiwire_gr (hwir_gr_test_ds)
- hiwire_sp (hwir_sp_test_ds)
- malorca (malorca_test_ds)
- nato (nato_test_ds)
- uwb (uwb_test_ds)

Očekává se, že datasety mají takové názvy, jaké byly vytvořené skripty pro tvorbu datasetů. Jsou uvedené vždy v závorce za názvem datasetu, který je má být použit jako parametr.


### ukázka spuštění eval.py
```
$ python eval.py \
  --metric wer \
  --datasets atco_en_ruzyne \
  --datasets_basedir ../data/atco/ \
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
  --NOP_wer_for_AG_existing_only \
  --use_prompt \
  --eval_callsigns \
```

## training.py
Skript využívaný pro trénování modelů. Parametry, které souvisí s evaluací jednotlivých epoch jsou nepoužité a označené slovem UNUSED (evaluace během trénování není implementována).

Parametry, které přijímá lze získat spuštěním `$ python training.py -h`
Názvy datasetů pro testování mohou být následující (v závorce je vždy uveden název složky, ve které je dataset očekáván):

- atco_test_en_ruzyne (en_ruzyne_test_ds)
- apimod (apimod_train_ds)
- atco_en (en_train_ds)
- atco_fr (fr_train_ds)
- hwir (hwir_train_ds)
- malorca (malorca_train_ds)
- nato (nato_train_ds)
- uwb (uwb_train_ds)

### ukázka spuštění training.py
```
$ python training.py \
  --model_path openai/whisper-tiny \
  --train_datasets atco_test_en_ruzyne \
  --datasets_root_dir ./data \
  --transcription_name_in_ds full_ts \
  --prompt_name_in_ds prompt_fullts_1G_4B \
  --output_dir ./test-ENCODER2 \
  --per_device_train_batch_size 4 \
  --gradient_accumulation_steps 4 \
  --learning_rate 1e-5 \
  --warmup_ratio 0.12 \
  --dropout 0.1 \
  --gradient_checkpointing \
  --fp16 \
  --save_strategy epoch \
  --num_train_epochs 10 \
  --per_device_eval_batch_size 4 \
  --predict_with_generate \
  --generation_max_length 448 \
  --logging_steps 30 \
  --report_to tensorboard 

```

## visual_eval_sentences.py
Skript použitý pro přepis daného datasetu. 
Parametry, které přijímá lze získat spuštěním `$ python visual_eval_sentences.py -h`

Názvy datasetů pro testování mohou být následující:
- atco_en_ruzyne (en_ruzyne_test_ds)
- atco_en_stefanik (en_stefanik_test_ds)
- atco_en_zurich (en_zurich_test_ds)
- atco_fr (fr_test_ds)
- atco_other_lang (other_lang_test_ds)
- hiwire_fr (hwir_fr_test_ds)
- hiwire_gr (hwir_gr_test_ds)
- hiwire_sp (hwir_sp_test_ds)
- malorca (malorca_test_ds)
- nato (nato_test_ds)
- uwb (uwb_test_ds)

Očekává se, že datasety mají takové názvy, jaké byly vytvořené skripty pro tvorbu datasetů. Jsou uvedené vždy v závorce za názvem datasetu, který je má být použit jako parametr.

### ukázka spuštění visual_eval_sentences.py
```
$ python scripts/visual_eval_sentences.py \
    --dataset atco_en_ruzyne atco_en_stefanik atco_en_zurich \
    --dataset_root data \
    --use_prompt \
    --model models/PROMPT/vanmed-full/AG/checkpoint-2772 \
    --output_file models/PROMPT/vanmed-full/exp/vis_ev_AG.txt \
    --prompt_name prompt_fullts_AG \
    --transcription_name full_ts 
```