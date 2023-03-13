"""Simple Dataset Reader"""

from typing import List, Union, Optional, Dict
from datasets import load_dataset
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer
from datasets.splits import NamedSplit
from openicl.icl_prompt_template import PromptTemplate
from openicl.utils.check_type import _check_dataset, _check_type_list, _check_str
import random
import torch


class DatasetReader:
    """In-conext Learning Dataset Reader Class
        Generate an DatasetReader instance through 'dataset'.
        
    Attributes:
        dataset (:obj:`Dataset` or :obj:`DatasetDict`): The dataset to be read.
        input_columns (:obj:`List[str]` or :obj:`str`): A list of column names (a string of column name) in the dataset that represent(s) the input field.
        output_column (:obj:`str`): A column name in the dataset that represents the prediction field.
        ds_size (:obj:`int` or :obj:`float`, optional): The number of pieces of data to return. When ds_size is an integer and greater than or equal to 1, `ds_size` pieces of data are randomly returned. When 0 < :obj:`ds_size` < 1, ``int(len(dataset) * ds_size)`` pieces of data are randomly returned. (used for testing)
        references(:obj:`list`, optional): The list of references, initialized by ``self.dataset[self.test_split][self.output_column]``.
        input_template (:obj:`PromptTemplate`, optional): An instance of the :obj:`PromptTemplate` class, used to format the input field content during the retrieval process. (in some retrieval methods)
        output_template (:obj:`PromptTemplate`, optional): An instance of the :obj:`PromptTemplate` class, used to format the output field content during the retrieval process. (in some learnable retrieval methods)
        input_output_template (:obj:`PromptTemplate`, optional): An instance of the `PromptTemplate` class, used to format the input-output field content during the retrieval process. (in some retrieval methods)
    """
    dataset = None
    input_template = None
    output_template = None
    input_output_template = None
    references = None

    def __init__(self,
                 dataset: Union[Dataset, DatasetDict, str],
                 input_columns: Union[List[str], str],
                 output_column: str,
                 name: Optional[str] = None,
                 data_files: Optional[str] = None,
                 input_template: Optional[PromptTemplate] = None,
                 output_template: Optional[PromptTemplate] = None,
                 input_output_template: Optional[PromptTemplate] = None,
                 ds_size: Union[None, int, float] = None,
                 split: Optional[NamedSplit] = None,
                 test_split: Optional[str] = 'test'
                 ) -> None:
        self.input_columns = _check_type_list(input_columns, [List, str])
        if isinstance(self.input_columns, str):
            self.input_columns = self.input_columns.split()
        self.output_column = _check_str(output_column)
        self.ds_size = _check_type_list(ds_size, [None, int, float])
        if input_template is not None:
            self.input_template = PromptTemplate._check_prompt_template(input_template)
        if output_template is not None:
            self.output_template = PromptTemplate._check_prompt_template(output_template)
        if input_output_template is not None:
            self.input_output_template = PromptTemplate._check_prompt_template(input_output_template)
        if isinstance(dataset, str):
            self.dataset = load_dataset(dataset, name=name, data_files=data_files)
        else:
            self.dataset = _check_dataset(dataset)
        if split is not None and isinstance(self.dataset, DatasetDict):
            self.dataset = self.dataset[split]
        if self.ds_size is not None:
            if isinstance(self.dataset, Dataset):
                self.dataset = load_partial_dataset(dataset, size=self.ds_size)
            if isinstance(self.dataset, DatasetDict):
                for ds_name in self.dataset.keys():
                    self.dataset[ds_name] = load_partial_dataset(self.dataset[ds_name], size=self.ds_size)
        if isinstance(self.dataset, DatasetDict):
            if test_split in self.dataset.keys():
                self.references = self.dataset[test_split][self.output_column]
        elif isinstance(self.dataset, Dataset):
            self.references = self.dataset[self.output_column]

    def set_references(self, column: str, split: Optional[str] = None) -> None:
        """Set :obj:`self.references` based on :obj:`column` and optional :obj:`split`.

        Args:
            column (:obj:`str`): A string of column name.
            split (:obj:`str`, optional): A string of dataset split. Defaults to ``None``.
        """
        if split is not None:
            self.references = self.dataset[split][column]
        else:
            self.references = self.dataset[column]

    def generate_input_field_prompt(self, entry: Dict) -> str:
        """Generate a prompt for the input field based on the provided :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the prompt.

        Returns:
            :obj:`str`: The generated prompt.
        """
        prompt = None
        if self.input_template is None:
            prompt = ' '.join([str(entry[ctx]) for ctx in self.input_columns])
        else:
            prompt = self.input_template.generate_item(entry)
        return prompt

    def generate_input_field_corpus(self, dataset: Union[Dataset, DatasetDict], split: Optional[str] = None) -> List[
        str]:
        """Generate corpus for input field.

        Args:
            dataset (:obj:`Dataset` or :obj:`DatasetDict`): A :obj:`datasets.Dataset` or :obj:`datasets.DatasetDict` instance.
            split (:obj:`str`, optional): The split of the dataset to use. If :obj:`None`, the entire dataset will be used. Defaults to ``None``.

        Returns:
            :obj:`List[str]`: A list of generated input field prompts.
        """
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_input_field_prompt(entry))
        return corpus

    def generate_ouput_field_prompt(self, entry: Dict) -> str:
        """Generate a prompt for the output field based on the provided :obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the prompt.

        Returns:
            :obj:`str`: The generated prompt.
        """
        prompt = None
        if self.output_template is None:
            prompt = str(entry[self.output_column])
        else:
            prompt = self.output_template.generate_item(entry)
        return prompt

    def generate_output_field_corpus(self, dataset: Union[Dataset, DatasetDict], split: Optional[str] = None) -> List[
        str]:
        """Generate corpus for output field.

        Args:
            dataset (:obj:`Dataset` or :obj:`DatasetDict`): A :obj:`datasets.Dataset` or :obj:`datasets.DatasetDict` instance.
            split (:obj:`str`, optional): The split of the dataset to use. If :obj:`None`, the entire dataset will be used. Defaults to ``None``.

        Returns:
            :obj:`List[str]`: A list of generated output field prompts.
        """
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_ouput_field_prompt(entry))
        return corpus

    def generate_input_output_field_prompt(self, entry: Dict) -> str:
        """Generate a prompt for the input-output field based on the provided:obj:`entry` data.

        Args:
            entry (:obj:`Dict`): A piece of data to be used for generating the prompt.

        Returns:
            :obj:`str`: The generated prompt.
        """
        prompt = None
        if self.input_output_template is None:
            prompt = ' '.join([entry[ctx] for ctx in self.input_columns] + [str(entry[self.output_column])])
        else:
            prompt = self.input_output_template.generate_item(entry)
        return prompt

    def generate_input_output_field_corpus(self, dataset: Union[Dataset, DatasetDict], split: Optional[str] = None) -> \
    List[str]:
        """Generate corpus for input-output field.

        Args:
            dataset (:obj:`Dataset` or :obj:`DatasetDict`): A :obj:`datasets.Dataset` or :obj:`datasets.DatasetDict` instance.
            split (:obj:`str`, optional): The split of the dataset to use. If :obj:`None`, the entire dataset will be used. Defaults to ``None``.

        Returns:
            :obj:`List[str]`: A list of generated input-output field prompts.
        """
        if split is not None:
            dataset = dataset[split]
        corpus = []
        for entry in dataset:
            corpus.append(self.generate_input_output_field_prompt(entry))
        return corpus

    def _check_dataset_reader(obj) -> "DatasetReader":
        if isinstance(obj, DatasetReader):
            return obj
        else:
            raise TypeError(f"Expected a DatasetReader object, but got {obj}")

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        return self.dataset[idx]

    def __repr__(self):
        return f"DatasetReader({{\n    dataset: {self.dataset},\n    input_columns: {self.input_columns},\n    output_columns: {self.output_column}\n}})"


def load_partial_dataset(dataset: Dataset, size: Optional[Union[int, float]] = None) -> Dataset:
    total_size = len(dataset)
    if size >= total_size or size <= 0:
        return dataset
    if size > 0 and size < 1:
        size = int(size * total_size)
    rand = random.Random(x=size)
    index_list = list(range(total_size))
    rand.shuffle(index_list)
    dataset = dataset.select(index_list[:size])
    return dataset


class DatasetEncoder(torch.utils.data.Dataset):
    def __init__(self, datalist: List, model_name=None, tokenizer=None) -> None:
        self.datalist = datalist
        if model_name is None and tokenizer is None:
            raise ValueError("model_name and tokenizer could not both be None")
        if tokenizer is not None:
            self.tokenizer = tokenizer
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.tokenizer.padding_side = "left"
        self.encode_dataset = []
        self.init_dataset()
        self.datalist_length = len(self.encode_dataset)

    def init_dataset(self):
        for idx, data in enumerate(self.datalist):
            tokenized_data = self.tokenizer.encode_plus(data, truncation=True, return_tensors='pt', verbose=False)
            self.encode_dataset.append({
                'input_ids': tokenized_data.input_ids[0],
                'attention_mask': tokenized_data.attention_mask[0],
                "metadata": {"id": idx, "len": len(tokenized_data.input_ids[0]),
                             "text": data}
            })

    def __len__(self):
        return self.datalist_length

    def __getitem__(self, idx):
        return self.encode_dataset[idx]
