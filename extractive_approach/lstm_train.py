import json
import os
import argparse

import numpy as np
import torch
import torch.nn as nn
from functools import partial
from torch.nn import LSTM
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm import tqdm
from lstm_model import MimicHospCourseTrainDataset, MimicHospCourseValDataset, MimicHospCourseTestDataset, LSTMClf, pad_train_sequence, pad_val_sequence, pad_test_sequence
from datasets import load_metric

os.environ['HF_DATASETS_CACHE'] = '/data/users/k1897038/.cache/huggingface/datasets'
os.environ['TRANSFORMERS_CACHE'] = '/data/users/k1897038/.cache/huggingface/transformers'

parser = argparse.ArgumentParser()

parser.add_argument('-sl', '--summ_lim', type=int, choices=(1, 2, 3, 5, 10, 15))
parser.add_argument('-cp', '--load_checkpoint', type=str, default=None)
parser.add_argument('-en', '--experiment_name', type=str)
parser.add_argument('-dsl', '--dataset_lim', type=int, default=-1)
parser.add_argument('-lr', '--learning_rate', type=float, default=0.01)
parser.add_argument('-e', '--epoch', type=int, default=5)
parser.add_argument('-bs', '--batch_size', type=int, default=50)
parser.add_argument('--use_sbert_embeddings', type=int, default=1)
parser.add_argument('--run_train', type=int, default=1)
parser.add_argument('--run_val', type=int, default=1)
parser.add_argument('--run_test', type=int, default=1)
parser.add_argument('--checkpoint_steps', type=int, default=200)
parser.add_argument('--cuda_device', type=str, default='0')
# lstm hiddnen layers,
# lstm bi-directional??

args = parser.parse_args()

gpu_num = args.cuda_device
device = f"cuda:{args.cuda_device}" if torch.cuda.is_available() else "cpu"

def main():
    sent_lim = args.summ_lim
    dataset_size_lim = args.dataset_lim
    bs = args.batch_size

    output_dir_base_path = '/data/users/k1897038/mimic_summarisation/extractive_approach'
    experiment_name = args.experiment_name
    experiment_path = f'{output_dir_base_path}/{experiment_name}'
    checkpoint_dir = f'{experiment_path}/checkpoints'
    outputs_dir = f'{experiment_path}/outputs'

    os.makedirs(checkpoint_dir, exist_ok=True)
    os.makedirs(outputs_dir, exist_ok=True)
    
        
    if args.use_sbert_embeddings:
        # default - use the sbert computed sentence emeddings
        input_prop_name = 'text_embed_limd'
        input_dim = 384
    else:
        # otherwise use spacy computed averaged GloVe sentence embeddings
        input_prop_name = 'text_embed_limd_spacy'
        input_dim = 300

    model = LSTMClf(input_dim=input_dim)
    criterion = nn.BCEWithLogitsLoss()
    optim = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    start_epoch = 0
    batch_skip = None

    checkpoint = args.load_checkpoint
    model.to(device)
    if checkpoint is not None:
        checkpoint = torch.load(checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optim.load_state_dict(checkpoint['optim_state_dict'])
        start_epoch = checkpoint['epoch']
        batch_skip = checkpoint.get('batch_skip')
        del checkpoint

    metric = load_metric('rouge')


    print("Loading train data...")
    train_ds = MimicHospCourseTrainDataset(sent_lim, size_lim=dataset_size_lim, input_prop_name=input_prop_name)
    print("Loading val data...")
    val_ds = MimicHospCourseValDataset(sent_lim, size_lim=dataset_size_lim, input_prop_name=input_prop_name)
    print("Loading test data...")
    test_ds = MimicHospCourseTestDataset(sent_lim, size_lim=dataset_size_lim, input_prop_name=input_prop_name)

    mimic_dl = partial(DataLoader, batch_size=bs, shuffle=False, pin_memory=True)
    train_loader = mimic_dl(train_ds, collate_fn=pad_train_sequence)
    val_loader = mimic_dl(val_ds, collate_fn=pad_val_sequence)
    test_loader = mimic_dl(test_ds, collate_fn=pad_test_sequence)

    # checkpoint every 200 batches
    checkpoint_after_steps = args.checkpoint_steps

    print('args run_train:' + str(args.run_train))
    if args.run_train:
        running_loss = []
        cum_val_loss = 0
        for epoch in range(start_epoch, args.epoch):
            print(f'Running training epoch:{epoch}')
            model.train()
            for i, (inputs, input_lens, outputs) in enumerate(tqdm(train_loader)):
                if batch_skip is not None and i < batch_skip:
                    continue
                inputs, outputs = inputs.to(device), outputs.to(device)
                logits = model(inputs, input_lens)
                loss = criterion(logits.squeeze(), outputs)
                loss.backward()
                running_loss.append(loss.item())
                parameters = filter(lambda p: p.requires_grad, model.parameters())
                torch.nn.utils.clip_grad_norm_(parameters, 0.25)
                optim.step()
                optim.zero_grad()
                if i % 20 == 0 and i != 0:
                    print(f'Loss: {loss}')
                if i % checkpoint_after_steps == 0 and i != 0:
                    print(f'Avg loss: {torch.mean(torch.tensor(running_loss))}')
                    save_checkpoint(checkpoint_dir, model, epoch, optim, batch_skip=i)
            batch_skip = None
            if args.run_val:
                print("computing val-set preds for end of epoch")
                cum_val_loss += run_val_set(model, val_loader, metric, criterion, outputs_dir, epoch)

    if bool(args.run_train) is False and args.run_val:
        print("computing val-set preds")
        run_val_set(model, val_loader, metric, criterion, outputs_dir, args.epoch)

    if args.run_test:
        print('Computing Test-set predictions')
        model.eval()
        for test_inputs, test_input_lens, input_sents, ref_sum_sents in tqdm(test_loader):
            # test results
            test_inputs = test_inputs.to(device)
            logits = model(test_inputs, test_input_lens)
            _preds_compute(metric, logits, input_sents, test_input_lens, ref_sum_sents)
        test_results = metric.compute()
        json.dump(test_results, open(f'{outputs_dir}/test-results.json', 'w'))
        print("saving test model")
        save_checkpoint(checkpoint_dir, model, args.epoch, optim, final=True)


def run_val_set(model, val_loader, metric, criterion, out_dir, epoch):
    model.eval()
    val_loss = 0
    for val_inputs, val_input_lens, outputs, input_sents, ref_sum_sents in tqdm(val_loader):
        # val loop
        val_inputs = val_inputs.to(device)
        outputs = outputs.to(device)
        logits = model(val_inputs, val_input_lens)
        loss = criterion(logits.squeeze(), outputs)
        val_loss += loss.cpu().item()
        _preds_compute(metric, logits, input_sents, val_input_lens, ref_sum_sents)
    val_results = metric.compute()
    json.dump(val_results,
              open(f'{out_dir}/val-results-epoch-{epoch}.json', 'w'))
    return val_loss / len(val_loader)


def save_checkpoint(checkpoint_dir, model, epoch, optim, batch_skip=None, final=False):
    path = f'{checkpoint_dir}/chk-epoch_{epoch}'
    path = path if batch_skip is None else path + f'-steps{batch_skip}'
    path = path if not final else path + '-final'
    path += '.pt'
    torch.save({
        'model_state_dict': model.state_dict(),
        'optim_state_dict': optim.state_dict(),
        'epoch': epoch,
        'batch_skip': batch_skip
    }, path)
    print(f'saving checkpoint: {path}')


def _preds_compute(metric, logits, input_sents, input_lens, ref_sums):
    summ_lim = args.summ_lim
    probs = torch.sigmoid(logits).squeeze()
    prob_ins = [prob[:in_len] for prob, in_len in zip(probs.detach().cpu().numpy(), input_lens)]
    # pull out top 'sum_limm' extractive sents
    pred_indxs = []
    for p in prob_ins:
        s_lim = len(p) if len(p) <= summ_lim else summ_lim
        pred_indxs.append(np.argpartition(p, -s_lim)[-s_lim:])
    pred_sums = []
    for sents, p_indxs in zip(input_sents, pred_indxs):
        pred_sums.append(''.join([sents[i] for i in p_indxs]))
    metric.add_batch(predictions=pred_sums, references=ref_sums)


if __name__ == '__main__':
    main()
