
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
import argparse, os, time
import json


class PrepareDatasetAsInput:

    def __init__(self, feature_extractor, tokenizer_en, tokenizer_fr, prompt_version=None):
        self.feature_extractor = feature_extractor
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr

    def set_transcription_name(self, transcription_name):
        self.transcription_name = transcription_name

    def set_prompt_name(self, prompt_name):
        self.prompt_name = prompt_name

    def prepare_dataset(self, batch):
        if (self.transcription_name is None):
            raise ValueError("Transcription name is not set. Please set it using set_transcription_name() method.")
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        features = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            truncation=False,
            padding='max_length',
            return_attention_mask=True
        )

        # add input features to batch
        batch["input_features"] = features.input_features[0]
        
        # add attention mask to batch, potentially for data augmentation
        batch['attention_mask'] = features.attention_mask[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        batch['labels'] = tokenizer(batch[self.transcription_name]).input_ids

        return batch

    def prepare_dataset_with_prompt(self,batch):
        if (self.transcription_name is None or self.prompt_name is None):
            raise ValueError("Transcription name is not set. Please set it using set_transcription_name() method.")
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        features = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            truncation=False,
            padding='max_length',
            return_attention_mask=True
        )

        # add input features to batch
        batch["input_features"] = features.input_features[0]
        
        # add attention mask to batch, potentially for data augmentation
        batch['attention_mask'] = features.attention_mask[0]

        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        # encode prompts to prompt ids - we assume that the dataset has a column `"prompt"` that contains the prompt for each example
        prompt_ids = tokenizer.get_prompt_ids(batch[self.prompt_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch["labels"] = prompt_ids + tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together

        return batch

    def prepare_dataset_self_prompt(self,batch):
        if (self.transcription_name is None):
            raise ValueError("Transcription name is not set. Please set it using set_transcription_name() method.")
        
        # load and resample audio data from 48 to 16kHz
        audio = batch["audio"]

        # compute log-Mel input features from input audio array
        features = self.feature_extractor(
            audio["array"], 
            sampling_rate=audio["sampling_rate"],
            truncation=False,
            padding='max_length',
            return_attention_mask=True
        )

        # add input features to batch
        batch["input_features"] = features.input_features[0]
        
        # add attention mask to batch, potentially for data augmentation
        batch['attention_mask'] = features.attention_mask[0]
        # encode target text to label ids **** CHANGED FROM **sentence** TO **transcription**
        # if french, than use french tokenizer, english otherwise
        tokenizer = self.tokenizer_en
        if "lang" in batch:
            if batch["lang"] == "fr":
                tokenizer = self.tokenizer_fr

        # make prompt from the lables
        prompt_ids = self.tokenizer_en.get_prompt_ids(batch[self.transcription_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch['labels'] = prompt_ids + tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together

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

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def compute_metrics_from_text(self,pred_str,label_str):

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return {"wer": wer}

    def compute_metrics_original(self, pred):
        pred_ids = pred.predictions
        label_ids = pred.label_ids

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

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

        # FOR USE ONLY IF WANT DATA AUGMENTATION
        batch['attention_mask'] = torch.tensor([mask['attention_mask'] for mask in features])

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
        
        # get the decoder input ids, by removing the last token (this is the 'shift' operation)
        decoder_input_ids = labels[:, :-1]

        # shift the labels to the left, to match work as prediction
        labels = labels[:, 1:]
        labels_mask = labels_batch.attention_mask[:, 1:]

        # replace padding with -100 to ignore correctly when computing the loss
        # if wondering why just labels_mask will have padding replaced by -100 and not decoder_input_ids
        # then it is because
        # only from labels loss is computed. If you dont pass decoder_input_ids and instead pass labels as is
        # done in the WOPrompt collator, then in shift_tokens_right function in modeling_whisper.py,
        # -100 are replaced by the pad_token_id, so it looks like for trainign pad_token_id is desired to be used
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
 
        # FOR USE ONLY IF WANT DATA AUGMENTATION
        batch['attention_mask'] = torch.tensor([mask['attention_mask'] for mask in features])
        
        return batch

@dataclass
class TrainingSetup:
    model_path: str
    train_datasets: list[str]
    datasets_root_dir: str
    transcription_name_in_ds : str
    prompt_name_in_ds : str
    dropout : float = 0.0
    eval_datasets: list[str] = None
    continue_from_checkpoint: bool = False
    use_prompt: bool = True
    self_prompt: bool = False
    freeze_encoder: bool = False

def build_eval_dataset(ds_list : list[str], prepare_dataset_fn, path_to_ds :str, separate_ds=False) -> dict[str,Dataset]|Dataset:  
    allds_test = {}
    # find all datasets to be tested
    for ds_name in ds_list:
        ds = None
        match ds_name:
            case 'atco_en_ruzyne':
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_ruzyne_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_ruzyne_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_ruzyne_test_ds"))
            case 'atco_en_stefanik':
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_stefanik_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_stefanik_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_stefanik_test_ds"))
            case 'atco_en_zurich':
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_zurich_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_zurich_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_zurich_test_ds"))
            case 'atco_fr':
                if (os.path.exists(os.path.join(path_to_ds,"atco/fr_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/fr_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"fr_test_ds"))
            case 'atco_other_lang':
                if (os.path.exists(os.path.join(path_to_ds,"atco/other_lang_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/other_lang_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"other_lang_test_ds"))
            case 'hiwire_fr':
                if (os.path.exists(os.path.join(path_to_ds,"hiwire/hwir_fr_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_fr_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"hwir_fr_test_ds"))
            case 'hiwire_gr':
                if (os.path.exists(os.path.join(path_to_ds,"hiwire/hwir_gr_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_gr_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"hwir_gr_test_ds"))
            case 'hiwire_sp':
                if (os.path.exists(os.path.join(path_to_ds,"hiwire/hwir_sp_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_sp_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"hwir_sp_test_ds"))
            case 'malorca':
                if (os.path.exists(os.path.join(path_to_ds,"malorca/malorca_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"malorca/malorca_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"malorca_test_ds"))
            case 'nato':
                if (os.path.exists(os.path.join(path_to_ds,"nato/nato_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"nato/nato_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"nato_test_ds"))
            case 'uwb':
                if (os.path.exists(os.path.join(path_to_ds,"uwb/uwb_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"uwb/uwb_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"uwb_test_ds"))
        if (ds):
            allds_test[ds_name]=ds
    
    # prepare the datasets to be ready for model
    for key in allds_test:
        allds_test[key] = allds_test[key].map(prepare_dataset_fn, remove_columns=ds.column_names, num_proc=1)
    
    # return either concatenated datasets or list of datasets
    if (separate_ds):
        return allds_test
    else:
        return concatenate_datasets([allds_test[key] for key in allds_test])

def build_train_dataset(list_of_ds : list[str], prepare_dataset_fn, path_to_ds :str, separate_ds=False) -> list[Dataset]|Dataset:
    allds_train = []
    for ds_name in list_of_ds:
        ds = None
        match ds_name:
            case 'atco_test_en_ruzyne': #TODO REMOVE
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_ruzyne_test_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_ruzyne_test_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_ruzyne_test_ds"))
            case 'apimod':
                if (os.path.exists(os.path.join(path_to_ds,"apimod/apimod_train_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"apimod/apimod_train_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"apimod_train_ds"))
            case 'atco_en':
                if (os.path.exists(os.path.join(path_to_ds,"atco/en_train_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/en_train_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"en_train_ds"))
            case 'atco_fr':
                if (os.path.exists(os.path.join(path_to_ds,"atco/fr_train_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"atco/fr_train_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"fr_train_ds"))
            case 'hwir':
                if (os.path.exists(os.path.join(path_to_ds,"hiwire/hwir_train_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"hiwire/hwir_train_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"hwir_train_ds"))
            case 'malorca':
                if (os.path.exists(os.path.join(path_to_ds,"malorca/malorca_train_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"malorca/malorca_train_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"malorca_train_ds"))
            case 'nato':
                if (os.path.exists(os.path.join(path_to_ds,"nato/nato_train_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"nato/nato_train_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"nato_train_ds"))
            case 'uwb':
                if (os.path.exists(os.path.join(path_to_ds,"uwb/uwb_train_ds"))):
                    ds = load_from_disk(os.path.join(path_to_ds,"uwb/uwb_train_ds"))
                else:
                    ds = load_from_disk(os.path.join(path_to_ds,"uwb_train_ds"))

        if (ds is not None):
            allds_train.append(ds)

    for idx,ds in enumerate(allds_train):
        allds_train[idx] = ds.map(prepare_dataset_fn, remove_columns=ds.column_names, num_proc=1)

    if (separate_ds):
        return allds_train
    else:
        return concatenate_datasets(allds_train)

def get_model_processor_tokenizerfr(model_path, training_setup : TrainingSetup) -> tuple[WhisperForConditionalGeneration, WhisperProcessor, WhisperTokenizer]:
    tokenizer_fr = WhisperTokenizer.from_pretrained(model_path, language="French", task="transcribe") # MODIF
    processor = WhisperProcessor.from_pretrained(model_path, language="English", task="transcribe")  # MODIF

    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    
    model.config.dropout = training_setup.dropout
    if (training_setup.freeze_encoder):
        model.freeze_encoder()
    
    # maybe this can be used for light data augmentation... maybe it can be enough
    # if you want to use this, you also need to uncomment lines in datacollators for obtaining attention_mask
    # model.config.apply_spec_augment = True
    # model.config.mask_time_prob = 0.05
    # model.config.mask_feature_prob = 0.05
    
    return model, processor, tokenizer_fr

def train(training_setup : TrainingSetup, training_args : Seq2SeqTrainingArguments):
    # get the model, processor and tokenizer
    model, processor, tokenizer_fr = get_model_processor_tokenizerfr(training_setup.model_path, training_setup)
    
    # load the dataset preparator, use prepare function according to prompt usage
    prepare_dataset = PrepareDatasetAsInput(processor.feature_extractor, processor.tokenizer, tokenizer_fr)
    prepare_dataset.set_transcription_name(training_setup.transcription_name_in_ds)
    prepare_dataset.set_prompt_name(training_setup.prompt_name_in_ds)

    # setup data PREPARATION FUNCTION and DATACOLLATOR according to prompt usage
    if training_setup.use_prompt:
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWITHPROMPT(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        if training_setup.self_prompt:
            prepare_fn = prepare_dataset.prepare_dataset_self_prompt
        else:
            prepare_fn = prepare_dataset.prepare_dataset_with_prompt
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWOPrompt(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id,
        )
        prepare_fn = prepare_dataset.prepare_dataset
        
    # load the train datasets
    train_ds = build_train_dataset(training_setup.train_datasets, prepare_fn, training_setup.datasets_root_dir, separate_ds=False)
    
    # load the eval datasets
    eval_ds = None
    if (training_setup.eval_datasets != None):
        eval_ds = build_eval_dataset(training_setup.eval_datasets, prepare_fn, training_setup.datasets_root_dir, separate_ds=False)

    # load the metric computer
    cm = ComputeMetrics(processor.tokenizer)
    
    trainer = Seq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=train_ds,
        data_collator=data_collator,
        eval_dataset=eval_ds,
        compute_metrics=cm.compute_metrics_original,
        processing_class=processor
    )
    
    print(f"Training start {' - resuming from checkpoint' if training_setup.continue_from_checkpoint else ''}...")
    
    if training_setup.continue_from_checkpoint:
        trainer.train(resume_from_checkpoint=training_setup.model_path)
    else:
        trainer.train()

def parse_args():
    parser = argparse.ArgumentParser(description="Parse training configuration")

    parser.add_argument('--setup', type=str, required=False, default=None, help='Path to the training setup')
    
    # training_setup
    parser.add_argument("--model_path", type=str, default="openai/whisper-tiny")
    parser.add_argument("--continue_from_checkpoint", action='store_true')
    parser.add_argument("--train_datasets", nargs='+', default=["atco_test_en_ruzyne"])
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--datasets_root_dir", type=str, default="./data")
    parser.add_argument("--use_prompt", action='store_true')
    parser.add_argument("--self_prompt", action='store_true')
    parser.add_argument("--transcription_name_in_ds", type=str, default="full_ts")
    parser.add_argument("--prompt_name_in_ds", type=str, default="prompt_fullts_1G_4B")
    parser.add_argument("--freeze_encoder", action='store_true')
    
    # training_args
    parser.add_argument("--output_dir", type=str, default="./test-nomask")
    parser.add_argument("--per_device_train_batch_size", type=int, default=4)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--learning_rate", type=float, default=1e-5)
    parser.add_argument("--warmup_ratio", type=float, default=0.12)
    parser.add_argument("--weight_decay", type=float, default=0.0)
    parser.add_argument("--gradient_checkpointing", action='store_true')
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--save_strategy", type=str, default="epoch")
    parser.add_argument("--num_train_epochs", type=int, default=10)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=4)
    parser.add_argument("--predict_with_generate", action='store_true')
    parser.add_argument("--generation_max_length", type=int, default=448)
    parser.add_argument("--logging_steps", type=int, default=30)
    parser.add_argument("--report_to", nargs='+', default=["tensorboard"])
    parser.add_argument("--metric_for_best_model", type=str, default="wer")
    parser.add_argument("--greater_is_better", action='store_true')
    parser.add_argument("--push_to_hub", action='store_true')

    return parser.parse_args()

def build_config(args):
    return {
        "training_setup": {
            "model_path": args.model_path,
            "continue_from_checkpoint": args.continue_from_checkpoint,
            "train_datasets": args.train_datasets,
            'dropout': args.dropout,
            "datasets_root_dir": args.datasets_root_dir,
            "use_prompt": args.use_prompt,
            "self_prompt": args.self_prompt,
            "transcription_name_in_ds": args.transcription_name_in_ds,
            "prompt_name_in_ds": args.prompt_name_in_ds,
            "freeze_encoder": args.freeze_encoder,
        },
        "training_args": {
            "output_dir": args.output_dir,
            "per_device_train_batch_size": args.per_device_train_batch_size,
            "gradient_accumulation_steps": args.gradient_accumulation_steps,
            "learning_rate": args.learning_rate,
            "warmup_ratio": args.warmup_ratio,
            "weight_decay": args.weight_decay,
            "gradient_checkpointing": args.gradient_checkpointing,
            "fp16": args.fp16,
            "save_strategy": args.save_strategy,
            "num_train_epochs": args.num_train_epochs,
            "per_device_eval_batch_size": args.per_device_eval_batch_size,
            "predict_with_generate": args.predict_with_generate,
            "generation_max_length": args.generation_max_length,
            "logging_steps": args.logging_steps,
            "report_to": args.report_to,
            "metric_for_best_model": args.metric_for_best_model,
            "greater_is_better": args.greater_is_better,
            "push_to_hub": args.push_to_hub
        }
    }
   
if __name__ == "__main__":
    # do parse args
    args = parse_args()
    
    # build the setup
    if (args.setup is None):
        # build the config from command line
        setup = build_config(args)
    else:    
        # load the setup
        with open(args.setup, 'r') as f:
            setup = json.load(f)
    
    # start the training setup
    training_setup = TrainingSetup(**setup['training_setup'])
    print(training_setup)

    # check if the model path exists and create the directory if not
    if not os.path.exists(setup['training_args']['output_dir']):
        os.makedirs(setup['training_args']['output_dir'])
    
    # save training details
    with open(os.path.join(setup['training_args']['output_dir'], "training_details.txt"), 'a') as f:        
        # print training setup
        f.write("=========================================================\n")
        f.write(f"**** TRAINING RUN, datetime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ****\n")
        if training_setup.continue_from_checkpoint:
            f.write(f" |* CONTINUE TRAINING FROM CHECKPOINT: {training_setup.model_path} *| \n")
        f.write("=========================================================\n")
        f.write("Training setup:\n")
        f.write(json.dumps(setup['training_setup'], indent=4))
        f.write('\n')
        f.write("Training arguments:\n")
        f.write(json.dumps(setup['training_args'], indent=4))
        f.write('\n')
        f.close()
    
    # setup trainign args
    training_args = Seq2SeqTrainingArguments(
        **setup['training_args']
    )
    
    # start training
    train(training_setup, training_args)

    # save info about finish at the end
    with open(os.path.join(setup['training_args']['output_dir'], "training_details.txt"), 'a') as f:
        f.write(f"Training finished, datetime: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())} ****\n")
        f.write("============================================\n")
        f.write("============================================\n\n")
        f.close()
        