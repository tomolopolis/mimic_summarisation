FROM tsearle/summ_exp:latest

#RUN jupyter server --generate-config

#RUN python download_hf_assets.py "seq2seq" "t5-small"
#RUN python download_hf_assets.py "seq2seq" "t5-base"
RUN python download_hf_assets.py "seq2seq" "facebook/bart-base"

RUN python download_hf_assets.py "seq2seq" "Kevincp560/distilbart-cnn-6-6-finetuned-pubmed"

RUN python download_hf_assets.py "seq2seq" "sshleifer/distilbart-xsum-9-6"

#
## did this load correctly?
#RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_ClinicalBERT"
#
#RUN python download_hf_assets.py "encoderDecoder" "emilyalsentzer/Bio_Discharge_Summary_BERT"

# can't load tokenizer?
#RUN python download_hf_assets.py "encoderDecoder"  "google/reformer-enwik8"

RUN python download_hf_assets.py "metric" "rouge"