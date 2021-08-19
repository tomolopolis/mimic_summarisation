import json

from datasets import load_metric
from transformers import AutoConfig, AutoTokenizer, AutoModelForSeq2SeqLM, AutoModel, EncoderDecoderConfig, \
    EncoderDecoderModel


def main():
    hf_cache_dir = './home/hf_cache_dir/'
    assets = json.load(open('hf_cache_config.json'))
    model_names = assets['models']['seq2seq']
    print(model_names)
    encoder_decoder_models = assets['models']['encoderDecoderModels']
    metrics = assets['metrics']
    use_fast_tokenizer = True

    for m_n in model_names:
        print(f'Downloading Seq2Seq HF model:{m_n}')
        config = AutoConfig.from_pretrained(m_n, cache_dir=hf_cache_dir)
        AutoTokenizer.from_pretrained(m_n, cache_dir=hf_cache_dir, use_fast=use_fast_tokenizer)
        AutoModelForSeq2SeqLM.from_pretrained(m_n, config=config, cache_dir=hf_cache_dir)

    for m_n in encoder_decoder_models:
        # https://huggingface.co/transformers/model_doc/encoderdecoder.html
        print(f'Downloading EncoderDecoder HF Model:{m_n}')
        AutoConfig.from_pretrained(m_n, cache_dir=hf_cache_dir)
        AutoTokenizer.from_pretrained(m_n, cache_dir=hf_cache_dir, use_fast=use_fast_tokenizer)
        EncoderDecoderModel.from_encoder_decoder_pretrained(m_n)

    for metric in metrics:
        print(f'Downloading HF metrics:{metric}')
        load_metric(metric)


if __name__ == '__main__':
    main()
