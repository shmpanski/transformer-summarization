import csv
import os

import numpy as np
import sentencepiece as spm
import torch


class DataLoader:
    def __init__(self, directory, parts, cols, spm_filename):
        """Dataset loader.

        Args:
            directory (str): dataset directory.
            parts (list[str]): dataset parts. [parts].tsv files must exists in dataset directory.
            spm_filename (str): file name of the dump sentencepiece model.

        """
        self.pad_idx, self.unk_idx, self.sos_idx, self.eos_idx = range(4)

        self.cols = cols
        self.directory = directory
        self.parts = parts
        self.spm_filename = spm_filename

        # Load sentecepiece model:
        self.sp = spm.SentencePieceProcessor()
        self.sp.load(self.spm_filename)

        # Load dataset parts:
        self.data_parts = {part: list(self.from_tsv(part)) for part in parts}
        self.part_lens = {part: len(self.data_parts[part]) for part in parts}
        self.max_lens = {part: self.get_max_len(part) for part in parts}
        self.max_len = max([self.max_lens[part] for part in parts])

    def next_batch(self, batch_size, part, device):
        """Get next batch.

        Args:
            batch_size (int): batch size.
            part (str): dataset part.
            device (torch.device): torch device.

        Returns:
            Batch: batch wrapper.

        """
        indexes = np.random.randint(0, self.part_lens[part], batch_size)
        raw_batches = [[self.data_parts[part][i][col] for i in indexes] for col, name in enumerate(self.cols)]

        return Batch(self, raw_batches, device)

    def sequential(self, part, device):
        """Get all examples from dataset sequential.

        Args:
            part (str): part of the dataset.
            device: (torch.Device): torch device.

        Returns:
            Batch: batch wrapper with size 1.

        """
        for example in self.data_parts[part]:
            raw_batches = [example]
            yield Batch(self, raw_batches, device)

    def pad(self, data):
        """Add <sos>, <eos> tags and pad sequences from batch

        Args:
           data (list[list[int]]): token indexes

        Returns:
            list[list[int]]: padded list of sizes (batch, max_seq_len + 2)
        """
        data = list(map(lambda x: [self.sos_idx] + x + [self.eos_idx], data))
        lens = [len(s) for s in data]
        max_len = max(lens)
        for i, length in enumerate(lens):
            to_add = max_len - length
            data[i] += [self.pad_idx] * to_add
        return data, lens

    def from_tsv(self, part):
        """Read and tokenize data from TSV file.

            Args:
                part (str): the name of the part.
            Yields:
                (list[int], list[int]): pairs for each example in dataset.

        """
        filename = os.path.join(self.directory, part + '.tsv')
        with open(filename) as file:
            reader = csv.reader(file, delimiter='\t')
            for row in reader:
                yield tuple(self.sp.EncodeAsIds(row[i]) for i, col in enumerate(self.cols))

    def decode(self, data):
        """Decode encoded sentence tensor.

        Args:
            data (torch.Tensor): sentence tensor.

        Returns:
            list[str]: decoded sentences.

        """
        return [self.sp.DecodeIds([token.item() for token in sentence]) for sentence in data]

    def decode_raw(self, data):
        """Decode encoded sentence tensor without removing auxiliary symbols.

                Args:
                    data (torch.Tensor): sentence tensor.

                Returns:
                    list[str]: decoded sentences.

                """
        return [''.join([self.sp.IdToPiece(token.item()) for token in sentence]) for sentence in data]

    def get_max_len(self, part):
        lens = []
        for example in self.data_parts[part]:
            for col in example:
                lens.append(len(col))
        return max(lens) + 2


class Batch:
    def __init__(self, data_loader, raw_batches, device):
        """Simple batch wrapper.

        Args:
            data_loader (DataLoader): data loader object.
            raw_batches (list[data]): raw data batches.
            device (torch.device): torch device.

        Variables:
            - **cols_name_length** (list[int]): lengths of `cols_name` sequences.
            - **cols_name** (torch.Tensor): long tensor of `cols_name` sequences.

        """
        for i, col in enumerate(data_loader.cols):
            tensor, length = data_loader.pad(raw_batches[i])
            self.__setattr__(col, torch.tensor(tensor, dtype=torch.long, device=device))
            self.__setattr__(col + '_length', length)

