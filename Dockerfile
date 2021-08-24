FROM tsearle/mimic_summ:base-no-model

ADD . /home/
WORKDIR /home/

RUN python download_hf_assets.py "metrics" "rouge"

RUN python download_hf_assets.py "seq2seq" "t5-small"

RUN python download_hf_assets.py "seq2seq" "t5-base"

RUN python download_hf_assets.py "seq2seq" "facebook/bart-base"

RUN python download_hf_assets.py "seq2seq" "google/bigbird-pegasus-large-pubmed",

RUN python download_hf_assets.py "seq2seq" "patrickvonplaten/led-large-16384-pubmed",

RUN python download_hf_assets.py "seq2seq" "microsoft/prophetnet-large-uncased"

RUN python download_hf_assets.py "encoderDecoder" "bert-base-cased",

RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_ClinicalBERT",

RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_Discharge_Summary_BERT",

RUN python download_hf_assets.py "encoderDecoder"  "google/reformer-enwik8"