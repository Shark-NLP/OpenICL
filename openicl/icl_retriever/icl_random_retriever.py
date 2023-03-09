'''Random Retriever'''

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger, SUBPROCESS_LOG_LEVEL
from typing import List, Union, Optional
from tqdm import trange
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)

class RandomRetriever(BaseRetriever):
    """Random In-context Learning Retriever Class
        Class of Random Retriever.
        
    Attributes:
        dataset_reader (DatasetReader): An instance of the `DatasetReader` class.
        ice_separator (str, optional): A string that separates each in-context example.
        ice_eos_token (str, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (str, optional): A string that is added to the end of the prompt.
        ice_num (int, optional): The number of data in the in-context examples.
        index_split (str, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. The default is 'train'.
        test_split (str, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. The default is 'test'.
        index_ds (Dataset): The index dataset. Used to select data for in-context examples.
        test_ds (Dataset): The test dataset. Used to generate prompts for each data.
        accelerator (Accelerator, optional): An instance of the `Accelerator` class, used for multiprocessing.
        seed (int, optional): Seed for the random number generator.
    """
    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] ='\n',
                 ice_eos_token: Optional[str] ='\n',
                 prompt_eos_token: Optional[str] = '',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 seed: Optional[int] = 43,
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split, test_split, accelerator)
        self.seed = seed
        if not self.is_main_process:
            logger.setLevel(SUBPROCESS_LOG_LEVEL)
        
        
    def retrieve(self):
        np.random.seed(self.seed)
        num_idx = len(self.index_ds)
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for _ in trange(len(self.test_ds), disable=not self.is_main_process):
            idx_list = np.random.choice(num_idx, self.ice_num, replace=False).tolist()
            rtr_idx_list.append(idx_list)
        return rtr_idx_list
        