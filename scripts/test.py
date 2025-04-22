import evaluate
import os
import numpy as np
import re
from transformers import WhisperTokenizer, WhisperProcessor

class ComputeMetrics:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.metric = evaluate.load("wer",experiment_id=str(os.getpid())+str(np.random.randint(0,10000000)))
        
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


class EvalCallsigns:
    wer_metric : ComputeMetrics
    def __init__(self, metric : ComputeMetrics):
        self.wer_metric = metric
    
    def __call__(self, transcriptions : list[str], callsigns : dict[str,int]) -> tuple[float, float, int]:
        """
        Args:
            transcription (str): The transcription of the audio.
            callsigns (dict[str,int]): A dictionary of callsigns and their number of occurences in the transcription.
        Returns:
            tuple[float, float, int]: A tuple containing the average WER, total WER, and the number of callsigns.
        """
        if (isinstance(callsigns,str)):
            callsigns = [{callsigns:1}]
        elif (isinstance(callsigns,dict)):
            callsigns = [callsigns]
            
        if (isinstance(transcriptions,str)):
            transcriptions = [transcriptions]
            
        total_wer = 0.0
        count_wer = 0
        completely_right = 0
        for transcription, callsigns in zip(transcriptions,callsigns):
            for callsign,num_of_occurences in callsigns.items():
                wer = self.find_lowest_wer(callsign, num_of_occurences, transcription)
                total_wer += sum(wer)
                count_wer += len(wer)
                completely_right += np.sum(np.where(np.array(wer) == 0,1,0))
                print({callsign:wer})
        
        return total_wer/count_wer, completely_right, total_wer, count_wer
    
    def find_lowest_wer(self, callsign : str, num_of_occurences : int, transcription : str) -> list[float]:
        callsign_norm = re.sub(r'\s+',' ',callsign.strip().lower()).split(' ')
        transcription_norm = re.sub(r'\s+',' ',transcription.strip().lower()).split(' ')
        # arange a list where wer will be stored
        wer_list = np.zeros(len(transcription_norm) - len(callsign_norm) + 1)
        # move a window with callsign through the transcription, compute wer and store
        for idx in range(0,len(transcription_norm) - len(callsign_norm) + 1):
            # check if the callsign is in the transcription
            cal_wer = self.wer_metric.compute_metrics_from_text(
                [' '.join(callsign_norm)], [' '.join(transcription_norm[idx:idx+len(callsign_norm)])]
            )
            wer_list[idx] = cal_wer

        # return as many lowest wer as num_of_occurences
        print(wer_list)
        return sorted(wer_list)[0:min(num_of_occurences,len(wer_list))]
    
    
if __name__ == "__main__":
    
    metric = ComputeMetrics(tokenizer=WhisperTokenizer.from_pretrained("openai/whisper-tiny"))
    eval_callsigns = EvalCallsigns(metric=metric)
    callsigns = [{
        "CSA One Delta Zulu": 3}]
    transcription = [
        " I asked if you want else to use, but..."]
    print(eval_callsigns(transcriptions=transcription, callsigns=callsigns))