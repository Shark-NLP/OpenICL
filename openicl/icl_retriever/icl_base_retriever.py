"""Basic Retriever"""

from datasets import Dataset, DatasetDict
from typing import List, Union, Optional, Tuple, Dict
from openicl import DatasetReader, PromptTemplate
from openicl.utils.check_type import _check_str
from accelerate import Accelerator


class BaseRetriever:
    """Basic In-context Learning Retriever Class
        Base class for In-context Learning Retriever, without any retrieval method.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
    """
    index_ds = None
    test_ds = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None
                 ) -> None:
        self.dataset_reader = DatasetReader._check_dataset_reader(dataset_reader)
        self.ice_separator = ice_separator
        self.ice_eos_token = ice_eos_token
        self.prompt_eos_token = prompt_eos_token
        self.ice_num = ice_num
        self.index_split = index_split
        self.test_split = test_split
        self.accelerator = accelerator
        self.is_main_process = True if self.accelerator is None or self.accelerator.is_main_process else False
        if isinstance(self.dataset_reader.dataset, Dataset):
            self.index_ds = self.dataset_reader.dataset
            self.test_ds = self.dataset_reader.dataset
            if self.accelerator is not None:
                self.test_ds = self.test_ds.shard(
                    num_shards=self.accelerator.num_processes,
                    index=self.accelerator.process_index
                )
        else:
            self.index_ds = self.dataset_reader.dataset[self.index_split]
            self.test_ds = self.dataset_reader.dataset[self.test_split]

            if self.accelerator is not None:
                self.test_ds = self.test_ds.shard(
                    num_shards=self.accelerator.num_processes,
                    index=self.accelerator.process_index
                )

    def retrieve(self) -> List[List]:
        """
            Retrieve for each data in generation_ds.
            
        Returns:
            `List[List]`: the index list of in-context example for each data in `test_ds`.
        """
        raise NotImplementedError("Method hasn't been implemented yet")

    def get_labels(self, ice_template: Optional[PromptTemplate] = None,
                   prompt_template: Optional[PromptTemplate] = None):
        labels = []
        if prompt_template is not None and isinstance(prompt_template.template, Dict):
            labels = list(prompt_template.template.keys())[:]
        elif ice_template is not None and ice_template.ice_token is not None and isinstance(ice_template.template,
                                                                                            Dict):
            labels = list(ice_template.template.keys())[:]
        else:
            labels = list(set(self.test_ds[self.dataset_reader.output_column]))
        return labels

    def generate_ice(self, idx_list: List[int], ice_template: Optional[PromptTemplate] = None) -> str:
        generated_ice_list = []
        dr = self.dataset_reader
        for idx in idx_list:
            if ice_template is None:
                generated_ice_list.append(' '.join(list(map(str,
                                                            [self.index_ds[idx][ctx] for ctx in dr.input_columns] + [
                                                                self.index_ds[idx][dr.output_column]]))))
            else:
                generated_ice_list.append(
                    ice_template.generate_ice_item(self.index_ds[idx], self.index_ds[idx][dr.output_column]))
        generated_ice = self.ice_separator.join(generated_ice_list) + self.ice_eos_token
        return generated_ice

    def generate_prompt(self, idx: int, ice: str, ice_template: Optional[PromptTemplate] = None,
                        prompt_template: Optional[PromptTemplate] = None) -> Tuple[List[str], List]:
        prompt_list = []
        labels = []
        if prompt_template is not None and isinstance(prompt_template.template, Dict):
            labels = list(prompt_template.template.keys())[:]
        elif ice_template is not None and isinstance(ice_template.template,
                                                     Dict) and ice_template.ice_token is not None:
            labels = list(ice_template.template.keys())[:]
        else:
            labels = list(set(self.test_ds[self.dataset_reader.output_column]))
        for label in labels:
            prompt_list.append(self.generate_label_prompt(idx, ice, label))
        return prompt_list, labels

    def generate_label_prompt(self, idx: int, ice: str, label, ice_template: Optional[PromptTemplate] = None,
                              prompt_template: Optional[PromptTemplate] = None, remain_sep: Optional[bool] = False) -> str:
        if prompt_template is not None:
            return prompt_template.generate_label_prompt_item(self.test_ds[idx], ice, label, remain_sep) + self.prompt_eos_token
        elif ice_template is not None and ice_template.ice_token is not None:
            return ice_template.generate_label_prompt_item(self.test_ds[idx], ice, label, remain_sep) + self.prompt_eos_token
        else:
            prefix_prompt = ' '.join(
                list(map(str, [self.test_ds[idx][ctx] for ctx in self.dataset_reader.input_columns])))
            return ice + prefix_prompt + ' ' + str(label) + self.prompt_eos_token

    def generate_prompt_for_generate_task(self, idx, ice, gen_field_replace_token='',
                                          ice_template: Optional[PromptTemplate] = None,
                                          prompt_template: Optional[PromptTemplate] = None):
        if prompt_template is not None:
            return prompt_template.generate_item(self.test_ds[idx], output_field=self.dataset_reader.output_column,
                                                 output_field_replace_token=gen_field_replace_token,
                                                 ice_field_replace_token=ice) + self.prompt_eos_token
        elif ice_template is not None and ice_template.ice_token is not None:
            return ice_template.generate_item(self.test_ds[idx], output_field=self.dataset_reader.output_column,
                                              output_field_replace_token=gen_field_replace_token,
                                              ice_field_replace_token=ice) + self.prompt_eos_token
        else:
            prefix_prompt = ' '.join(
                list(map(str, [self.test_ds[idx][ctx] for ctx in self.dataset_reader.input_columns])))
            return ice + prefix_prompt + gen_field_replace_token + self.prompt_eos_token
