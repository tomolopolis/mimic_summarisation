FROM tsearle/mimic_summ:base-no-model

RUN python download_hf_assets.py "metrics" "rouge"

RUN python download_hf_assets.py "seq2seq" "t5-small"

RUN python download_hf_assets.py "seq2seq" "t5-base"

RUN python download_hf_assets.py "seq2seq" "facebook/bart-base"

# doesn't seem to load in hf
#RUN python download_hf_assets.py "seq2seq" "google/bigbird-pegasus-large-pubmed"

RUN python download_hf_assets.py "seq2seq" "patrickvonplaten/led-large-16384-pubmed"

RUN python download_hf_assets.py "seq2seq" "microsoft/prophetnet-large-uncased"

# did this load correctly?
RUN python download_hf_assets.py "encoderDecoder" "bert-base-cased"

# did this load correctly?
RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_ClinicalBERT"

RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_Discharge_Summary_BERT"

# can't load tokenizer?
#RUN python download_hf_assets.py "encoderDecoder"  "google/reformer-enwik8"

RUN python download_hf_assets.py "metric" "rouge"