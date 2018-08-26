import argparse
import logging
import os

import sentencepiece as spm
from gensim.models import Word2Vec

from data.utils import SequentialSentenceLoader, export_embeddings

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

description = 'Preprocessor for summarization dataset. Include unsupervised text tokenization and vector word' \
              'representation using word2vec model.' \
              'Dataset must consists of two parts: train and test stored in `train.tsv` and `test.tsv` respectively. ' \
              'After training embedding model is saving into dataset directory.'

parser = argparse.ArgumentParser(description=description, formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('--dataset', metavar='DIR', type=str, default='./dataset', help='dataset directory')
parser.add_argument('--vocab_size', metavar='V', type=int, default=25000, help='vocabulary size')
parser.add_argument('--emb_size', metavar='E', type=int, default=250, help='embedding size')
parser.add_argument('--workers', metavar='WS', type=int, default=3, help='number of cpu cores, uses for training')
parser.add_argument('--sg', action='store_true', help='use skip-gram for training word2vec')
parser.add_argument('--prefix', metavar='P', type=str, default='simple-summ', help='model prefix')

args = parser.parse_args()

train_filename = os.path.join(args.dataset, 'train.tsv')
test_filename = os.path.join(args.dataset, 'test.tsv')
sp_model_prefix = os.path.join('./models_dumps', args.prefix, args.prefix + '_bpe')
sp_model_filename = sp_model_prefix + '.model'
w2v_model_filename = os.path.join('./models_dumps', args.prefix, 'word2vec.model')
embeddings_filename = os.path.join('./models_dumps', args.prefix, 'embedding.npy')

for path in [sp_model_filename, w2v_model_filename, embeddings_filename]:
    os.makedirs(os.path.dirname(path), exist_ok=True)

# Start tokenization training:
spm_params = '--pad_id=0 --unk_id=1 --bos_id=2 --eos_id=3 ' \
             '--input={} --model_prefix={} --vocab_size={}'.format(train_filename, sp_model_prefix, args.vocab_size)
spm.SentencePieceTrainer.Train(spm_params)

# Load trained sentencepice model:
sp = spm.SentencePieceProcessor()
sp.load(sp_model_filename)

# Next, train word2vec embeddings:
sentences = SequentialSentenceLoader(train_filename, sp)
w2v_model = Word2Vec(sentences, min_count=0, workers=args.workers, size=args.emb_size, sg=int(args.sg))
w2v_model.save(w2v_model_filename)

# Export embeddings into lookup table:
export_embeddings(embeddings_filename, sp, w2v_model)
logging.info('Embeddings have been saved into {}'.format(embeddings_filename))
