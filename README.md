# PyTorch Attentive Summarization NN
## Model description
Summarization model for short texts based on pure [transformer model](https://arxiv.org/abs/1706.03762) with [bpe encoding](http://www.aclweb.org/anthology/P16-1162).

## Requirements
* Python 3.5 or higher
* PyTorch 0.4.0 or higher
* sentencepiece
* gensim
* tensorboardX

## Usage
### Preprocessing
First of all, each part of dataset must stores in common folder and .tsv files, e.g. `./dataset/train.tsv` and `./dataset/test.tsv` (`./dataset/sample.tsv` by default uses for sampling summarizations). Each line of the document has two parts: *source* and  *target* separated by `tab` symbol. Tabular files don't need headers.  Preprocessor build vocabulary and train word embeddings.

For preprocess use:
```
$ python preprocess.py
```

### Training
For train model use
```
$ python train.py --cuda --pretrain_emb
```
After training model is saving into `./models_dumps/` directory.

You can tune model with lots arguments available in model.
If you want a good result, use **much deeper** configuration of the model! Default configuration is used for tests.

### Sampling
For generate summarizations use
```
$ python sample.py --inp=sample_part --out=output_file.txt
```

Where `sample_part` is dataset part (e.g. `./dataset/sample_part.tsv`).

**See help of each module for more information and available arguments.**