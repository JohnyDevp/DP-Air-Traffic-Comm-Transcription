## write out from whisper finetune article
- whisper is sequence to sequence model. It maps swquence of audio spectrogram features to a sequence of text tokens.
1. the raw audio inputs are converted to a log-mel spectrogram. **feature extractor**
2. transformer encoder encodes the spectrogram to form a sequence of encoder hidden states.
3. decoder autoregressively predicts text tokens, conditional on both the previous tokens and the encoder hidden states
- process of *incorporating a language model* internally in the system architecture is termed **deep fusion**
(contrast to *shallow fusion* where a lang. model is combined externally with an encoder, such as with CTC + n-gram)
    - with deep fusion the entire system can be trained end-to-end with same training data and loss. resulting in greater flexibility and generally superior performance
4. whisper is pre-trained and fine-tuned usisng cross entropy (classic ...)
- whisper has several configurations resulting in different model sizes, the largest is multilingual only (rest also english-only) ... english-only, multilingual-only means on which data it was trained
- The name Whisper follows from the acronym “WSPSR”, which stands for “Web-scale Supervised Pre-training for Speech Recognition”.
- whisper is unique in the point, that there is no requirement for an attention mask when forwarding audio samples to the whisper model, since all inputs are padded/truncated to the length of 30s - the MAXIMUM INPUT LENGTH. This is difference of most other audio models, where you need to provide an attention mask with details where the sequnces has been padded and thus where they should be ignored in the self-attention mechanism. whisper infers directly from the speech signals where to ignore inputs.
- The Whisper model outputs text tokens that indicate the index of the predicted text among the dictionary of vocabulary items. The tokenizer maps a sequence of text tokens to the actual text string (e.g. [1169, 3797, 3332] -> "the cat sat").

1. padding/truncating signal to 30s -> log mel spectrogram
## next tips for writing 
- mention CTC (Connectionist Temporal Classification), it is used for encoder-only models for ASR...
- log mel cepstrum, process of creation of it,
- log mel spectrogram, what is speech
- common voice data, what is it, how you will be using it 
- three stages of your training
    1. finetuining the model on given data by Szoke
    2. finetuning on common voice data
    3. prompt driven finetuning

