FROM pytorch/pytorch:1.4-cuda10.1-cudnn7-runtime
#FROM deepspeed/deepspeed:latest

ADD . /home/

WORKDIR /home/

RUN pip install -r requirements.txt

RUN git clone https://github.com/huggingface/transformers.git

RUN pip install -e transformers/.

RUN python -c "import nltk; nltk.download('punkt', quiet=True)"
