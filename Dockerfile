FROM tsearle/mimic_summ:base-no-model

ADD . /home/
WORKDIR /home/

RUN python download_hf_assets.py "seq2seq" "t5-small"


