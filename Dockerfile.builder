FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime

ADD . /home/

WORKDIR /home/

RUN pip install -r requirements.txt

RUN git clone https://github.com/huggingface/transformers.git

RUN pip install -e transformers/.

RUN chmod u+x jupyter_start.sh && chmod u+x train_baseline.sh

RUN python -c "import nltk; nltk.download('punkt', quiet=True)"
