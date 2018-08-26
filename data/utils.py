import os

import numpy as np


class SequentialSentenceLoader:
    """Load tokenized sentences from file.

    Args:
        filename (str): the name of the input file.
        sp_model (sentencepiece.SentencePieceProcessor): a sentencepice model.

    Yields:
        list[str]: tokens from sentence.
    """

    def __init__(self, filename, sp_model):
        self.filename = filename
        self.sp_model = sp_model

    def __iter__(self):
        with open(self.filename, 'r') as file:
            for line in file:
                for sentence in line.split('\t'):
                    encoded_sentence = list(map(str, self.sp_model.EncodeAsIds(sentence)))
                    yield ['2'] + encoded_sentence + ['3']


def export_embeddings(filename, sp_model, w2v_model):
    """Export embeddings into numpy matrix.

    Args:
        filename (str): the name of the exported file.
        sp_model (sentencepice.SentencePieceProcessor): Sentencepice model.
        w2v_model (gensim.models.Word2Vec): Word2Vec model.
    """
    dim = w2v_model.vector_size
    vocab_size = len(sp_model)
    table = np.array([w2v_model[str(i)] if str(i) in w2v_model.wv else np.zeros([dim]) for i in range(vocab_size)])
    np.save(filename, table)


def import_embeddings(filename):
    """Import embeddings from dump file.

    Args:
        filename: the name of the embedding dump file.

    Returns:
        numpy.ndarray: embedding matrix.
    """
    return np.load(filename)

