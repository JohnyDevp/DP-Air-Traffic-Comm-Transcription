
### author
#It is original whisper-finetune script, taken from [colab](https://colab.research.google.com/github/sanchit-gandhi/notebooks/blob/main/fine_tune_whisper.ipynb), and adapted for my special task.

# IMPORTS

import torch
import numpy as np
from dataclasses import dataclass
import evaluate
from transformers import WhisperProcessor, WhisperTokenizer, Seq2SeqTrainingArguments, WhisperForConditionalGeneration, Seq2SeqTrainer
from datasets import Dataset, load_from_disk, concatenate_datasets

from typing import Any, Dict, List, Union
import argparse, os
import json

class PrepareDatasetAsInput:
    
    def __init__(self, feature_extractor, tokenizer_en, tokenizer_fr):
        self.feature_extractor = feature_extractor
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
            
    def prepare_dataset(self, batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        batch["labels_fullts"] = tokenizer(batch["full_ts"]).input_ids
        batch["labels_shortts"] = tokenizer(batch["short_ts"]).input_ids

        return batch
    
    def prepare_dataset_with_prompt(self,batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode prompts to prompt ids - we assume that the dataset has a column `"prompt"` that contains the prompt for each example
        prompt_ids = []
        if 'prompt' in batch:
            prompt_ids = self.tokenizer_en.get_prompt_ids(batch["prompt"]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        token_ids = tokenizer(batch["full_ts"]).input_ids

        batch["labels_fullts"] = prompt_ids + token_ids # building labels ids with prompt and tokens together
        batch["labels_shortts"] = tokenizer(batch["short_ts"]).input_ids

        return batch
    
    def prepare_dataset_self_prompt(self,batch):
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        batch["input_features"] = self.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr
        
        # make prompt from the lables
        fullts_prompt_ids = self.tokenizer_en.get_prompt_ids(batch["full_ts"]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines
        shortts_prompt_ids = self.tokenizer_en.get_prompt_ids(batch["short_ts"]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines
        
        batch["labels_fullts"] = fullts_prompt_ids + tokenizer(batch["full_ts"]).input_ids # building labels ids with prompt and tokens together
        batch["labels_shortts"] = shortts_prompt_ids + tokenizer(batch["short_ts"]).input_ids

        return batch

class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer")
        
    def compute_metrics(self,pred_text, reference_text):
        pred_ids = pred_text
        label_ids = reference_text

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}
    
    def compute_metrics_from_text(self,pred_str,label_str):

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingWOPrompt:
    processor: Any
    decoder_start_token_id: int

    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # replace padding with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        return batch

@dataclass
class DataCollatorSpeechSeq2SeqWithPaddingWITHPROMPT:
    processor : Any
    decoder_start_token_id: int
    def __call__(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # ==================================================================================
        # WORKING CALL, USED IN PROMPT-TEST-3A, CORRECT, ONLY WITH 448 MAX LENGTH
        # ==================================================================================

        # split inputs and labels since they have to be of different lengths and need
        # different padding methods

        # dataloader returns a list of features which we convert to a dict
        input_features = [{"input_features": feature["input_features"]} for feature in features]
        label_features = [{"input_ids": feature["labels"]} for feature in features]

        # reformat list to dict and set to pytorch format
        batch = self.processor.feature_extractor.pad(
            input_features,
            return_tensors="pt",
        )

        labels_batch = self.processor.tokenizer.pad(
            label_features,
            return_tensors="pt",
        )

        # shift labels to the right to get decoder input ids
        labels = labels_batch["input_ids"]

        decoder_input_ids = labels[:, :-1]
        
        # shift the labels to the left, to match work as prediction
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids

        return batch
    
    def myoldcall_stillworking(self, features: List[Dict[str, Union[List[int], torch.Tensor]]]) -> Dict[str, torch.Tensor]:
        # split inputs and labels since they have to be of different lengths and need different padding methods
        # first treat the audio inputs by simply returning torch tensors

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")

        # get the tokenized label sequences
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        # pad the labels to max length
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")

        # copy the labels as in its form now (with both prompt and the transcript itself) it should be the input to the decoder
        batch['decoder_input_ids'] = labels_batch["input_ids"].clone()[:, :-1]

        # replace padding in labels with -100 to ignore loss correctly
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)[:, 1:]
        
        # then mask out the prompt in the labels
        bos_index = np.argmax(labels==self.decoder_start_token_id, axis=1)
        prompt_mask = np.arange(labels.shape[1]) < bos_index.numpy()[:, np.newaxis]
        labels = torch.tensor(np.where(prompt_mask, -100, labels))

        # if bos token is appended in previous tokenization step,
        # cut bos token here as it's append later anyways (actually it's not needed to cut it here, because it the bos
        # tokeon is appended just if decode_input_ids is not provided)
        # STILL WE LET IT BE THERE AS PART OF ORIGINAL CODE
        if (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
            labels = labels[:, 1:]

        batch["labels"] = labels

        # and at last we create attention mask, to tell the decoder to use the correct part from the labels
        # attention_mask = torch.tensor(np.where(decoder_input_ids != tokenizer_en.pad_token_id, 1 , 0))
        # batch["attention_mask"] = attention_mask

        return batch

@dataclass
class TrainingSetup:
    model_path: str
    continue_from_checkpoint: bool
    train_datasets: list[str]
    path_to_train_datasets: str
    use_prompt: bool
    self_prompt: bool
    
def build_dataset(list_of_ds : list[str], prepare_dataset_fn, path_to_ds :str, separate_ds=False, ts='fullts') -> list[Dataset]|Dataset:
    allds_train = []
    for ds_name in list_of_ds:
        ds = None
        match ds_name:
            case 'apimod':
                ds = load_from_disk(os.path.join(path_to_ds,"apimod/apimod_train_ds"))
            case 'atco_en':
                ds = load_from_disk(os.path.join(path_to_ds,"atco/en_train_ds"))
            case 'atco_fr':
                ds = load_from_disk(os.path.join(path_to_ds,"atco/fr_train_ds"))
            case 'hwir':
                ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_train_ds"))
            case 'malorca':
                ds = load_from_disk(os.path.join(path_to_ds,"malorca/malorca_train_ds"))
            case 'nato':
                ds = load_from_disk(os.path.join(path_to_ds,"nato/nato_train_ds"))
            case 'uwb':
                ds = load_from_disk(os.path.join(path_to_ds,"uwb/uwb_train_ds"))
        if (ds is not None):
            allds_train.append(ds)

    for idx,ds in enumerate(allds_train):
        allds_train[idx] = ds.map(prepare_dataset_fn, remove_columns=ds.column_names, num_proc=1)
        allds_train[idx] = allds_train[idx].rename_column('labels_fullts' if ts == 'fullts' else 'labels_shortts','labels')
    
    if (separate_ds):
        return allds_train
    else:
        return concatenate_datasets(allds_train)

def get_model_processor_tokenizerfr(model_path) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer]:
    tokenizer_fr = WhisperTokenizer.from_pretrained(model_path, language="french", task="transcribe")
    processor = WhisperProcessor.from_pretrained(model_path, language="English", task="transcribe")
    
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None

    return model, processor, tokenizer_fr

if __name__ == "__main__":
    # do parse args
    parser = argparse.ArgumentParser(description='Evaluate transcription accuracy in WER or CER.')
    parser.add_argument('--setup', type=str, required=True, help='Path to the training setup')
    args = parser.parse_args()
    
    # load the setup
    with open(args.setup, 'r') as f:
        setup = json.load(f)
    training_setup = TrainingSetup(**setup['training_setup'])

    # get the model, processor and tokenizer
    model, processor, tokenizer_fr = get_model_processor_tokenizerfr(training_setup.model_path)
    
    # load the dataset preparator, use prepare function according to prompt usage
    prepare_dataset = PrepareDatasetAsInput(processor.feature_extractor, processor.tokenizer, tokenizer_fr)
    if training_setup.use_prompt:
        if training_setup.self_prompt:
            prepare_fn = prepare_dataset.prepare_dataset_self_prompt
        else:
            prepare_fn = prepare_dataset.prepare_dataset_with_prompt
    else:
        prepare_fn = prepare_dataset.prepare_dataset
        
    # load the datasets
    train_ds = build_dataset(training_setup.train_datasets, prepare_fn, training_setup.path_to_train_datasets, separate_ds=False)
    
    # load the data collator according to prompt usage
    if training_setup.use_prompt:
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWITHPROMPT(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWOPrompt(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )

    # load the metric computer
    cm = ComputeMetrics(processor.tokenizer)
    
    # Extract training arguments explicitly from the setup JSON
    training_args = Seq2SeqTrainingArguments(
        **setup['training_args']
    )
    
    # training_args = Seq2SeqTrainingArguments(
    #     output_dir="./drive/Shareddrives/DP/models/train-prototype",
    #     per_device_train_batch_size=4,
    #     gradient_accumulation_steps=4,
    #     learning_rate=1e-5,
    #     warmup_ratio=0.12,
    #     gradient_checkpointing=True,
    #     fp16=True,
    #     # eval_strategy="epoch", # changed from evaluation_strategy (because of warning)
    #     save_strategy="epoch",
    #     num_train_epochs=2,
    #     per_device_eval_batch_size=8,
    #     predict_with_generate=True,
    #     generation_max_length=448, # 448 MAX LENGTH AVAILABLE, DO NOT CHANGE !!!
    #     logging_steps=30,
    #     report_to=["tensorboard"],
    #     # load_best_model_at_end=True,
    #     metric_for_best_model="wer",
    #     greater_is_better=False,
    #     push_to_hub=False, # change to True to push the model to the Hub (need to be logged in)
    # )

    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        # eval_dataset=ds_dict["test"],
        data_collator=data_collator,
        compute_metrics=cm.compute_metrics,
        processing_class=processor
    )

    if training_setup.continue_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_setup.model_path)
    else:
        trainer.train()

    # trainer.evaluate()