from io import FileIO, TextIOWrapper
import json
from click import File, prompt
from h11 import Data
from regex import D
from sklearn.linear_model import OrthogonalMatchingPursuitCV
import torch
from torch.utils.data import DataLoader
from jiwer import wer, cer
import os 
import argparse
from datasets import load_from_disk, concatenate_datasets, Dataset
from typing import Any, Dict, List, Union
from dataclasses import dataclass
from transformers import WhisperTokenizer, WhisperProcessor, WhisperForConditionalGeneration
import evaluate
from tqdm import tqdm
import numpy as np


class PrepareDatasetAsInput:
    
    def __init__(self, feature_extractor, tokenizer_en, tokenizer_fr):
        self.feature_extractor = feature_extractor
        self.tokenizer_en = tokenizer_en
        self.tokenizer_fr = tokenizer_fr
    
    def set_transcription_name(self, transcription_name):
        self.transcription_name = transcription_name
    
    def set_prompt_name(self, prompt_name):
        self.prompt_name = prompt_name
               
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

        batch['labels'] = tokenizer(batch[self.transcription_name]).input_ids

        return batch
    
    def prepare_dataset_with_prompt(self,batch):
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
        
        # encode prompts to prompt ids - we assume that the dataset has a column `"prompt"` that contains the prompt for each example
        prompt_ids = tokenizer.get_prompt_ids(batch[self.prompt_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch["prompt_ids"] = prompt_ids
        batch["labels"] = tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together
        
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
        prompt_ids = self.tokenizer_en.get_prompt_ids(batch[self.transcription_name]).tolist() # YOU NEED TO ADD TOLIST() because array cant be combined with list in the next lines

        batch['prompt_ids'] = prompt_ids 
        batch['labels'] = tokenizer(batch[self.transcription_name]).input_ids # building labels ids with prompt and tokens together

        return batch
    
class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer")
        
    def compute_metrics(self,pred_text, reference_text) -> float:
        pred_ids = pred_text
        label_ids = reference_text

        # replace -100 with the pad_token_id
        label_ids[label_ids == -100] = self.tokenizer.pad_token_id

        # we do not want to group tokens when computing the metrics
        pred_str = self.tokenizer.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = self.tokenizer.batch_decode(label_ids, skip_special_tokens=True)

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return wer
    
    def compute_metrics_from_text(self,pred_str,label_str) -> float:

        wer = 100 * self.metric.compute(predictions=pred_str, references=label_str)

        return wer

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
        prompt_features = [{'prompt_ids': feature['prompt_ids']} for feature in features]
        
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
        labels = labels.masked_fill(labels_mask.ne(1), -100)

        # replace initial prompt tokens with -100 to ignore correctly when computing the loss
        bos_index = torch.argmax((labels == self.decoder_start_token_id).long(), dim=1)
        prompt_mask = torch.arange(labels.shape[1]) < bos_index[:, None]
        labels = torch.where(prompt_mask, -100, labels)

        batch["labels"] = labels
        batch["decoder_input_ids"] = decoder_input_ids
        
        # added especially for EVALUATION
        batch['prompt_ids'] = prompt_features
        
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
class EvaluationSetup:
    r"""
        Class that stores the evalution parameters passed to script
        
        `same_processor` : bool

    """
    metric : str
    datasets : str
    datasets_basedir : str
    models : str | list
    output_file : str
    transcription_name_in_ds : str
    batch_size : int = 1
    same_processor : bool = True
    prompt_name_in_ds : str = None
    eval_description : str = ""
    overwrite : bool = False
    separate_ds : bool = False
    use_prompt : bool = True
    self_prompt : bool = False


from rapidfuzz import process
class EvalCallsigns:
    
    def __obtain_callsign_from_transcription(self, callsigns, callsigns_pos, transcription : str):
        ts_arr = transcription.strip().lower().split(' ')
        for callsign in callsigns:
            callsign_arr = callsign.strip().lower().split(' ')    
            # check if the callsign is in the transcription
            for cal in callsign_arr:
                process.extractOne(cal, ts_arr, scorer=80)
    
    def __search_for_callsign(self, callsign):
        # search for the callsign in the dataset
        # if found, return True
        # if not found, return False
        pass

def build_dataset(ds_list : list[str], prepare_dataset_fn, path_to_ds :str, separate_ds=False) -> dict[str,Dataset]|Dataset:  
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

def compute(test_ds : dict[str,Dataset]|Dataset, model, processor, metric, batch_size=3, use_prompt=False) -> dict[str,dict]:        
    # define collator
    if (not use_prompt):
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWOPrompt(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    else:
        data_collator = DataCollatorSpeechSeq2SeqWithPaddingWITHPROMPT(
            processor=processor,
            decoder_start_token_id=model.config.decoder_start_token_id
        )
    
    # run the evaluation
    if (isinstance(test_ds, Dataset) and not use_prompt):
        # setup the dataloader
        dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)
        
        # Iterate over batches
        all_preds = []
        all_lables = []
        all_loss = []
        for batch in tqdm(dataloader):
            input_features = batch["input_features"].cuda()
            
            preds = model.generate(input_features).detach().cpu()
            outputs = model(input_features, labels=batch["labels"].cuda())
            
            all_loss.append(outputs.loss.detach().cpu())
            all_preds.extend(processor.batch_decode(preds, skip_special_tokens=True))
            all_lables.extend(processor.batch_decode(batch["labels"], skip_special_tokens=True))

        
        # compute the wer
        wer=metric.compute_metrics_from_text(all_preds, all_lables)   
        loss=torch.mean(torch.tensor(all_loss,dtype=float))
        
        return {'allds':{'wer':wer,'loss':loss}}
    
    elif (isinstance(test_ds, dict) and not use_prompt):
        out = {}
        for ds_name, ds in test_ds.items():
            # setup the dataloader
            dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)
            
            # Iterate over batches
            all_preds = []
            all_lables = []
            all_loss = []
            for batch in tqdm(dataloader):
                input_features = batch["input_features"].cuda()
                
                preds = model.generate(input_features).detach().cpu()
                outputs = model(input_features, labels=batch["labels"].cuda())
                
                all_loss.append(outputs.loss.detach().cpu())
                all_preds.extend(processor.batch_decode(preds, skip_special_tokens=True))
                all_lables.extend(processor.batch_decode(batch["labels"], skip_special_tokens=True))

            
            # compute the wer
            wer=metric.compute_metrics_from_text(all_preds, all_lables)   
            loss=torch.mean(torch.tensor(all_loss,dtype=float))
            
            out[ds_name] = {'wer':wer,'loss':loss}

        return out

    elif (isinstance(test_ds, Dataset) and use_prompt):
        # WHEN using PROMPT, so far we can test only one sample at a time with one promp
        # because yet we cannot handle different prompts for different samples in the same batch
        # setup the dataloader
        dataloader = DataLoader(test_ds, batch_size=batch_size, collate_fn=data_collator)
   
        
        # Iterate over batches
        all_preds = []
        all_lables = []
        all_loss = []
        for batch in tqdm(dataloader):
            
            # pass everything to CUDA
            for key in batch:
                if (key == 'prompt_ids'): continue
                batch[key] = batch[key].cuda()
            
            # loop through the batch and compute the predicitons
            # it is necessary because model.generate() cannot handle multiple input features with multiple prompt 
            for idx in range(batch_size):
                # check whether the index doesnt go above the number of items in current batch
                if (idx >= len(batch['input_features'])): break
                
                input_features = batch["input_features"][idx].unsqueeze(0)
                prompt_ids = torch.tensor(batch["prompt_ids"][idx]['prompt_ids']).cuda()
                preds = model.generate(input_features, prompt_ids=prompt_ids).detach().cpu()
                
                # compute loss together for all batch
                all_preds.extend([processor.tokenizer.decode(preds[0], skip_special_tokens=True)])
                all_lables.extend([processor.tokenizer.decode(batch["labels"][idx], skip_special_tokens=True)])
            
            # compute the loss
            with torch.no_grad():
                outputs = model(input_features = batch['input_features'], labels=batch['labels'], decoder_input_ids=batch['decoder_input_ids'])
            # append the loss result to the whole result obtained so far
            all_loss.append(outputs.loss.detach().cpu())
            
        # compute the wer
        wer=metric.compute_metrics_from_text(all_preds, all_lables)   
        loss = torch.mean(torch.tensor(all_loss),dtype=float)
        return {'allds':{'wer':wer, 'loss':loss}}
    elif (isinstance(test_ds, dict) and use_prompt):
        # WHEN using PROMPT, so far we can test only one sample at a time with one promp
        # because yet we cannot handle different prompts for different samples in the same batch
        # setup the dataloader
        out = {}
        for ds_name,ds in test_ds.items():
            dataloader = DataLoader(ds, batch_size=batch_size, collate_fn=data_collator)
    
            # Iterate over batches
            all_preds = []
            all_lables = []
            all_loss = []
            for batch in tqdm(dataloader):
                
                # pass everything to CUDA
                for key in batch:
                    if (key == 'prompt_ids'): continue
                    batch[key] = batch[key].cuda()
                
                # loop through the batch and compute the predicitons
                # it is necessary because model.generate() cannot handle multiple input features with multiple prompt 
                for idx in range(batch_size):
                    # check whether the index doesnt go above the number of items in current batch
                    if (idx >= len(batch['input_features'])): break
                    
                    input_features = batch["input_features"][idx].unsqueeze(0)
                    prompt_ids = torch.tensor(batch["prompt_ids"][idx]['prompt_ids']).cuda()
                    preds = model.generate(input_features, prompt_ids=prompt_ids).detach().cpu()
                    
                    # compute loss together for all batch
                    all_preds.extend([processor.tokenizer.decode(preds[0], skip_special_tokens=True)])
                    all_lables.extend([processor.tokenizer.decode(batch["labels"][idx], skip_special_tokens=True)])
                
                # compute the loss
                with torch.no_grad():
                    outputs = model(input_features = batch['input_features'], labels=batch['labels'], decoder_input_ids=batch['decoder_input_ids'])
                # append the loss result to the whole result obtained so far
                all_loss.append(outputs.loss.detach().cpu())
                
            # compute the wer and loss for current dataset
            wer=metric.compute_metrics_from_text(all_preds, all_lables)   
            loss = torch.mean(torch.tensor(all_loss),dtype=float)
            out[ds_name] = {'wer':wer, 'loss':loss}
            
        return out
    else:
        raise ValueError('Wrong dataset format')

def setup_model_processor(model_path) -> tuple[WhisperForConditionalGeneration, WhisperProcessor]:
    # setup the model
    model = WhisperForConditionalGeneration.from_pretrained(model_path)
    if torch.cuda.is_available():
        model.cuda()
    # setting the parameters of model for correct working
    model.generation_config.language = "english"
    model.generation_config.task = "transcribe"
    model.generation_config.forced_decoder_ids = None
    processor = WhisperProcessor.from_pretrained(model_path)
    
    return model, processor

def result_printout(f_desc : TextIOWrapper, out_dict : dict[str, dict], evaluation_setup : EvaluationSetup):
    for k,v in out_dict.items():
        print(f'DATASET: {k} | WER: {v['wer']} LOSS: {v['loss']}')
        
    # write the output
    f_desc.write("******** Evaluation setup ********\n")
    f_desc.write(evaluation_setup.__str__())
    f_desc.write("\n")
    f_desc.write("******** Evaluation description ********\n")
    f_desc.write(evaluation_setup.eval_description)
    f_desc.write("\n")
    f_desc.write("******** Evaluation results ********\n")
    for k,v in out_dict.items():
        f_desc.write(f'DATASET: {k} | WER: {v['wer']} LOSS: {v['loss']}')
           
def main(evaluation_setup : EvaluationSetup):
    if (evaluation_setup.metric == 'wer'):
        # handle the output file
        if (evaluation_setup.overwrite):
            out_file= open(evaluation_setup.output_file, 'w')
        else: 
            out_file= open(evaluation_setup.output_file, 'a')
        
        # handle multiple models 
        if (isinstance(evaluation_setup.models,list)):       
            # go through each model, clear the memory and see the results
            first_model_in_serie = True
            
            # these setups can change over iterations through the list of models
            # when models are same, than this setup is stored and reused 
            prep_ds_cls : PrepareDatasetAsInput     = None
            built_ds : dict[Dataset]|Dataset        = None
            metric : ComputeMetrics                 = None
            tokenizer_fr : WhisperTokenizer         = None
                        
            for model_path in evaluation_setup.models:
                out_file.write(f'#### EVAL MODEL {model_path} ####\n')
                print(f'*** Model path {model_path} ***')
                
                model, processor = setup_model_processor(model_path)

                if (not evaluation_setup.same_processor or first_model_in_serie):
                    tokenizer_fr = WhisperTokenizer.from_pretrained(model_path, language="French") # just for the french tokenizer
                    # above are the only conditions when new data parsing is necessary
                    # prepare new handler for dataset
                    prep_ds_cls = PrepareDatasetAsInput(processor.feature_extractor, processor.tokenizer, tokenizer_fr)
                    # set up the prompt and transcription names in the given dataset, that will be used
                    prep_ds_cls.set_transcription_name(evaluation_setup.transcription_name_in_ds)
                    prep_ds_cls.set_prompt_name(evaluation_setup.prompt_name_in_ds)    
                    
                    # choose function for dataset parsing
                    if evaluation_setup.use_prompt:
                        if evaluation_setup.self_prompt:
                            prepare_fn = prep_ds_cls.prepare_dataset_self_prompt
                        else:
                            prepare_fn = prep_ds_cls.prepare_dataset_with_prompt
                    else:
                        prepare_fn = prep_ds_cls.prepare_dataset
                        
                    built_ds = build_dataset(
                        evaluation_setup.datasets, 
                        prepare_fn,
                        evaluation_setup.datasets_basedir, 
                        evaluation_setup.separate_ds
                    )
                    
                    metric = ComputeMetrics(processor.tokenizer)
                    
                # compute the wer and loss
                out_dict = compute(built_ds, model, processor, metric, batch_size=3, use_prompt=evaluation_setup.use_prompt)
                
                # print results to the file
                result_printout(out_file,out_dict,evaluation_setup)
                
                # new lines at the end of the file
                out_file.write('\n\n')
                
                if first_model_in_serie:
                    first_model_in_serie = False 
            
            # close file 
            out_file.close()
        else:
            # run the evaluation of single model
            # ==========================================                    
                        
            # setup model and the processor
            model, processor = setup_model_processor(evaluation_setup.models)
            tokenizer_fr = WhisperTokenizer.from_pretrained(evaluation_setup.models, language="French") # just for the french tokenizer
            prep_ds_cls = PrepareDatasetAsInput(processor.feature_extractor, processor.tokenizer, tokenizer_fr)
            
            # set up the prompt and transcription names in the given dataset, that will be used
            prep_ds_cls.set_transcription_name(evaluation_setup.transcription_name_in_ds)
            prep_ds_cls.set_prompt_name(evaluation_setup.prompt_name_in_ds)
            
            ds_list = evaluation_setup.datasets
            
            if evaluation_setup.use_prompt:
                if evaluation_setup.self_prompt:
                    prepare_fn = prep_ds_cls.prepare_dataset_self_prompt
                else:
                    prepare_fn = prep_ds_cls.prepare_dataset_with_prompt
            else:
                prepare_fn = prep_ds_cls.prepare_dataset
            built_ds = build_dataset(
                ds_list, 
                prepare_fn,
                evaluation_setup.datasets_basedir, 
                evaluation_setup.separate_ds
            )
            
            metric = ComputeMetrics(processor.tokenizer)
            
            # compute the wer and loss
            out_dict = compute(built_ds, model, processor, metric, batch_size=3, use_prompt=evaluation_setup.use_prompt)
            
            result_printout(out_file, out_dict, evaluation_setup)
            
            out_file.close()
            
    elif (evaluation_setup.metric == 'cer'):
        metric = cer
    
if __name__ == '__main__':
    # parse args
    parser = argparse.ArgumentParser(description='Evaluate transcription accuracy in WER or CER.')
    parser.add_argument('--setup', type=str, required=True, help='Type of evaluation metric')
    evaluation_setup = parser.parse_args()
    
    # load the setup
    with open(evaluation_setup.setup, 'r') as f:
        setup = json.load(f)
    evaluation_setup = EvaluationSetup(**setup)
    
    print('******** Evaluation setup ********')
    print(evaluation_setup.__str__())
    
    # run the evaluation
    main(evaluation_setup)