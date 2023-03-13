"""BM25 Retriever"""

from openicl import DatasetReader
from openicl.icl_retriever import BaseRetriever
from openicl.utils.logging import get_logger
from typing import List, Union, Optional
from rank_bm25 import BM25Okapi
import numpy as np
from tqdm import trange
from accelerate import Accelerator
from nltk.tokenize import word_tokenize

logger = get_logger(__name__)


class BM25Retriever(BaseRetriever):
    """BM25 In-context Learning Retriever Class
        Class of BM25 Retriever.
        
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
        index_corpus (:obj:`List[str]`) : A corpus created from the input field data of :obj:`index_ds`.
        test_corpus (:obj:`List[str]`) : A corpus created from the input field data of :obj:`test_ds`.
        bm25 (:obj:`BM250kapi`): An instance of :obj:`BM250kapi` class, initialized using :obj:`index_ds`.
    """
    bm25 = None
    index_corpus = None
    test_corpus = None

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
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, ice_num, index_split,
                         test_split, accelerator)
        self.index_corpus = [word_tokenize(data) for data in
                             self.dataset_reader.generate_input_field_corpus(self.index_ds)]
        self.bm25 = BM25Okapi(self.index_corpus)
        self.test_corpus = [word_tokenize(data) for data in
                            self.dataset_reader.generate_input_field_corpus(self.test_ds)]

    def retrieve(self) -> List[List]:
        rtr_idx_list = []
        logger.info("Retrieving data for test set...")
        for idx in trange(len(self.test_corpus), disable=not self.is_main_process):
            query = self.test_corpus[idx]
            scores = self.bm25.get_scores(query)
            near_ids = list(np.argsort(scores)[::-1][:self.ice_num])
            near_ids = [int(a) for a in near_ids]
            rtr_idx_list.append(near_ids)
        return rtr_idx_list
