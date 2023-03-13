"""Votek Retriever"""

import os
import json
from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from typing import List, Union, Optional, Tuple
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import numpy as np
import random
from accelerate import Accelerator


class VotekRetriever(TopkRetriever):
    """Vote-k In-context Learning Retriever Class
        Class of Vote-k Retriever.
        
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
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:``model``.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
        votek_k (:obj:`int`, optional): ``k`` value of Voke-k Selective Annotation Algorithm. Defaults to ``3``.
    """

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 votek_k: Optional[int] = 3,
                 accelerator: Optional[Accelerator] = None,
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token,
                         sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size,
                         accelerator)
        self.votek_k = votek_k

    def votek_select(self, embeddings=None, select_num=None, k=None, overlap_threshold=None, vote_file=None):
        n = len(embeddings)
        if vote_file is not None and os.path.isfile(vote_file):
            with open(vote_file) as f:
                vote_stat = json.load(f)
        else:
            vote_stat = defaultdict(list)

            for i in range(n):
                cur_emb = embeddings[i].reshape(1, -1)
                cur_scores = np.sum(cosine_similarity(embeddings, cur_emb), axis=1)
                sorted_indices = np.argsort(cur_scores).tolist()[-k - 1:-1]
                for idx in sorted_indices:
                    if idx != i:
                        vote_stat[idx].append(i)

            if vote_file is not None:
                with open(vote_file, 'w') as f:
                    json.dump(vote_stat, f)
        votes = sorted(vote_stat.items(), key=lambda x: len(x[1]), reverse=True)
        j = 0
        selected_indices = []
        while len(selected_indices) < select_num and j < len(votes):
            candidate_set = set(votes[j][1])
            flag = True
            for pre in range(j):
                cur_set = set(votes[pre][1])
                if len(candidate_set.intersection(cur_set)) >= overlap_threshold * len(candidate_set):
                    flag = False
                    break
            if not flag:
                j += 1
                continue
            selected_indices.append(int(votes[j][0]))
            j += 1
        if len(selected_indices) < select_num:
            unselected_indices = []
            cur_num = len(selected_indices)
            for i in range(n):
                if not i in selected_indices:
                    unselected_indices.append(i)
            selected_indices += random.sample(unselected_indices, select_num - cur_num)
        return selected_indices

    def vote_k_search(self):
        vote_k_idxs = self.votek_select(embeddings=self.embed_list, select_num=self.ice_num, k=self.votek_k,
                                        overlap_threshold=1)
        return [vote_k_idxs[:] for _ in range(len(self.test_ds))]

    def retrieve(self):
        return self.vote_k_search()
