"""Zeroshot Retriever"""

from datasets import Dataset, DatasetDict
from typing import List, Union, Optional, Tuple, Dict
from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever import BaseRetriever
from openicl.utils.check_type import _check_str
from accelerate import Accelerator

class ZeroRetriever(BaseRetriever):
    """Zero In-context Learning Retriever Class
        Retriever for Zero-shot.
        
    Attributes:
        dataset_reader (DatasetReader): An instance of the `DatasetReader` class.
        ice_eos_token (str, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (str, optional): A string that is added to the end of the prompt.
        index_split (str, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. The default is 'train'.
        test_split (str, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. The default is 'test'.
        index_ds (Dataset): The index dataset. Used to select data for in-context examples.
        test_ds (Dataset): The test dataset. Used to generate prompts for each data.
        accelerator (Accelerator, optional): An instance of the `Accelerator` class, used for multiprocessing.
    """    
    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_eos_token: Optional[str] ='',
                 prompt_eos_token: Optional[str] = '',
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__(dataset_reader, '', ice_eos_token, prompt_eos_token, 0, index_split, test_split, accelerator)

    def retrieve(self) -> List[List]:
        rtr_idx_list = [[] for _ in range(len(self.test_ds))]
        return rtr_idx_list
    