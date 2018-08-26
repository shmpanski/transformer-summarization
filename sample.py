import argparse
import os
import pickle

import torch

from data import DataLoader
from model import TransformerSummarizer

description = 'Utility for sampling summarization.'

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--inp', metavar='I', type=str, default='sample', help='name sample part of dataset')
parser.add_argument('--out', metavar='O', type=str, default='./dataset/generated.txt', help='output file')
parser.add_argument('--prefix', metavar='P', type=str, default='simple-summ', help='model prefix')
parser.add_argument('--dataset', metavar='D', type=str, default='./dataset', help='dataset folder')
parser.add_argument('--limit', metavar='L', type=int, default=15, help='generation limit')

args = parser.parse_args()

bpe_model_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '_bpe.model')
model_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '.model')
model_args_filename = os.path.join('./models_dumps', args.prefix, args.prefix + '.args')
emb_filename = os.path.join('./models_dumps', args.prefix, 'embedding.npy')

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

loader = DataLoader(args.dataset, [args.inp], ['src'], bpe_model_filename)

args_m = pickle.load(open(model_args_filename, 'rb'))

model = TransformerSummarizer(**args_m)

model.load_state_dict(torch.load(model_filename))
model.to(device)
model.eval()

with torch.no_grad():
    summ = []
    for batch in loader.sequential(args.inp, device):
        seq = model.sample(batch, args.limit)
        summ += loader.decode(seq)
    with open(args.out, 'w') as f:
        f.write('\n'.join(summ))
