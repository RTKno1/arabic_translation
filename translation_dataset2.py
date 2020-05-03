import os
import xml.etree.ElementTree as ET
import glob
import io
import codecs

#from torch.utils import data
from torchtext import data


class TranslationDataset2(data.Dataset):
    """Defines a dataset for machine translation."""

    @staticmethod
    def sort_key(ex):
        return data.interleave_keys(len(ex.src), len(ex.trg))

    def __init__(self, path, exts, **kwargs):
        """Create a TranslationDataset given paths and fields.

        Arguments:
            path: Common prefix of paths to the data files for both languages.
            exts: A tuple containing the extension to path for each language.
            Remaining keyword arguments: Passed to the constructor of
                data.Dataset.
        """

        src_path, trg_path = tuple(os.path.expanduser(path + x) for x in exts)
        
        examples = []
        with io.open(src_path, mode='r', encoding='utf-8') as src_file, \
                io.open(trg_path, mode='r', encoding='utf-8') as trg_file:
            for src_line, trg_line in zip(src_file, trg_file):
                src_line, trg_line = src_line.strip(), trg_line.strip()
                if src_line != '' and trg_line != '':
                    examples.append(data.Example.fromlist(
                        [src_line, trg_line]))
    
    
    @classmethod
    def splits(cls, exts, path=None, root='.data',
               train='train', validation='val', test='test', **kwargs):
        """Create dataset objects for splits of a TranslationDataset.

        Arguments:
            exts: A tuple containing the extension to path for each language.
            fields: A tuple containing the fields that will be used for data
                in each language.
            path (str): Common prefix of the splits' file paths, or None to use
                the result of cls.download(root).
            root: Root dataset storage directory. Default is '.data'.
            train: The prefix of the train data. Default: 'train'.
            validation: The prefix of the validation data. Default: 'val'.
            test: The prefix of the test data. Default: 'test'.
            Remaining keyword arguments: Passed to the splits method of
                Dataset.
        """
        if path is None:
            path = cls.download(root)

        train_data = None if train is None else cls(
            os.path.join(path, train), exts, **kwargs)
        val_data = None if validation is None else cls(
            os.path.join(path, validation), exts, **kwargs)
        test_data = None if test is None else cls(
            os.path.join(path, test), exts, **kwargs)
        return tuple(d for d in (train_data, val_data, test_data)
                     if d is not None)

        
        
        
        

#tests 

eng_msa_train_dataset = TranslationDataset2(path='data/train/train_eng_msa.', exts=('eng', 'msa'))
