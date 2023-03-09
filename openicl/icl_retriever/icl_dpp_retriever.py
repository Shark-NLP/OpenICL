"""DPP Retriever"""

from openicl import DatasetReader
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.logging import get_logger, SUBPROCESS_LOG_LEVEL
from typing import Optional
from dppy.finite_dpps import FiniteDPP
import tqdm
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)

class DPPRetriever(TopkRetriever):
    """DPP In-context Learning Retriever Class
        Class of DPP Retriever.
        
    Attributes:
        dataset_reader (DatasetReader): An instance of the `DatasetReader` class.
        ice_separator (str, optional): A string that separates each in-context example.
        ice_eos_token (str, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (str, optional): A string that is added to the end of the prompt.
        ice_num (int, optional): The number of data in the in-context examples.
        candidate_num (int, optional): The number of data selected in TopK stage.
        index_split (str, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. The default is 'train'.
        test_split (str, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. The default is 'test'.
        index_ds (Dataset): The index dataset. Used to select data for in-context examples.
        test_ds (Dataset): The test dataset. Used to generate prompts for each data.
        accelerator (Accelerator, optional): An instance of the `Accelerator` class, used for multiprocessing.
        batch_size (int, optional): Batch size for the `DataLoader`. 
        model (SentenceTransformer): An instance of `SentenceTransformer` class, used to calculate embeddings.
        tokenizer (AutoTokenizer): Tokenizer for `model`.
        index (IndexIDMap): Index generated with FAISS.
        seed (int, optional): Seed for the random number generator. (`random_state` in `sample_exact_k_dpp` method)
        scale_factor (float, optional): A factor when gets the kernel.
    """
    model = None
    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] ='\n',
                 ice_eos_token: Optional[str] ='\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name : Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 candidate_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 seed: Optional[int] = 1,
                 scale_factor: Optional[float] = 0.1
    ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size, accelerator) 
        if not self.is_main_process:
            logger.setLevel(SUBPROCESS_LOG_LEVEL)
        self.candidate_num = candidate_num
        self.seed = seed
        self.scale_factor = scale_factor
        
    
    def dpp_search(self):
        res_list = self.forward(self.dataloader, process_bar=True, information="Embedding test set...")
        rtr_idx_list = [[] for _ in range(len(res_list))]
        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']
            
            # get results of TopK
            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = np.array(self.index.search(embed, self.candidate_num)[1][0].tolist())
            
            # DPP stage
            near_reps, rel_scores, kernel_matrix = self.get_kernel(embed, near_ids.tolist())
            dpp_L = FiniteDPP('likelihood', **{'L': kernel_matrix})
            
            entry_legal_flag = False
            seed = self.seed
            while entry_legal_flag != True:
                try:
                    samples_ids = np.array(dpp_L.sample_exact_k_dpp(size=self.ice_num, random_state=self.seed))
                except:
                    seed = seed + 1
                    logger.warning(f'illegal seed {seed} for this entry (processing test_set data {idx}), trying seed {seed + 1}')
                    if (seed > 9999999):
                        raise RuntimeError('Endless loop')
                    continue
                entry_legal_flag = True
            
            
            # ordered by relevance score
            samples_scores = np.array([rel_scores[i] for i in samples_ids])
            samples_ids = samples_ids[(-samples_scores).argsort()].tolist()
            rtr_sub_list = [int(near_ids[i]) for i in samples_ids]
            
            rtr_idx_list[idx] = rtr_sub_list

        return rtr_idx_list
            
            
    def retrieve(self):
        return self.dpp_search()

    def get_kernel(self, embed, candidates):
        near_reps = np.stack([self.index.index.reconstruct(i) for i in candidates], axis=0)
        # normalize first
        embed = embed / np.linalg.norm(embed)
        near_reps = near_reps / np.linalg.norm(near_reps, keepdims=True, axis=1)

        rel_scores = np.matmul(embed, near_reps.T)[0]
        rel_scores = (rel_scores + 1) / 2
        
        # to balance relevance and diversity
        rel_scores = np.exp(rel_scores / (2 * self.scale_factor))

        sim_matrix = np.matmul(near_reps, near_reps.T)
        sim_matrix = (sim_matrix + 1) / 2
        
        kernel_matrix = rel_scores[None] * sim_matrix * rel_scores[:, None]
        return near_reps, rel_scores, kernel_matrix
