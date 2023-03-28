"""MDL Retriever"""

from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever.icl_topk_retriever import TopkRetriever
from openicl.utils.calculate import entropy
from openicl.utils.logging import get_logger
from typing import List, Union, Optional, Tuple
from transformers import AutoModelForCausalLM
import tqdm
import torch
import numpy as np
from accelerate import Accelerator

logger = get_logger(__name__)


class MDLRetriever(TopkRetriever):
    """MDL In-context Learning Retriever Class
        Class of MDL Retriever.
        
    Attributes:
        dataset_reader (:obj:`DatasetReader`): An instance of the :obj:`DatasetReader` class.
        ice_separator (:obj:`str`, optional): A string that separates each in-context example.
        ice_eos_token (:obj:`str`, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (:obj:`str`, optional): A string that is added to the end of the prompt.
        ice_num (:obj:`int`, optional): The number of data in the in-context examples.
        candidate_num (:obj:`int`, optional): The number of data selected in TopK stage.
        index_split (:obj:`str`, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. Defaults to ``train``.
        test_split (:obj:`str`, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. Defaults to ``test``.
        index_ds (:obj:`Dataset`): The index dataset. Used to select data for in-context examples.
        test_ds (:obj:`Dataset`): The test dataset. Used to generate prompts for each data.
        accelerator (:obj:`Accelerator`, optional): An instance of the :obj:`Accelerator` class, used for multiprocessing.
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        model (:obj:`SentenceTransformer`): An instance of :obj:`SentenceTransformer` class, used to calculate embeddings.
        tokenizer (:obj:`AutoTokenizer`): Tokenizer for :obj:`model`.
        index (:obj:`IndexIDMap`): Index generated with FAISS.
        select_time (:obj:`int`, optional): Number of random selections in the MDL stage.
        labels (:obj:`List`, optional): A list of labels for all classes used to generate prompts when calculating MDL.
        seed (:obj:`int`, optional): Seed for the random number generator.
    """
    metric_model = None

    def __init__(self,
                 dataset_reader: DatasetReader,
                 ice_separator: Optional[str] = '\n',
                 ice_eos_token: Optional[str] = '\n',
                 prompt_eos_token: Optional[str] = '',
                 sentence_transformers_model_name: Optional[str] = 'all-mpnet-base-v2',
                 ice_num: Optional[int] = 1,
                 candidate_num: Optional[int] = 1,
                 index_split: Optional[str] = 'train',
                 test_split: Optional[str] = 'test',
                 tokenizer_name: Optional[str] = 'gpt2-xl',
                 ce_model_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 select_time: Optional[int] = 5,
                 accelerator: Optional[Accelerator] = None,
                 ice_template: Optional[PromptTemplate] = None,
                 prompt_template: Optional[PromptTemplate] = None,
                 labels: Optional[List] = None,
                 seed: Optional[int] = 1
                 ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token,
                         sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size,
                         accelerator)
        self.ce_model_name = ce_model_name
        self.candidate_num = candidate_num
        self.select_time = select_time
        self.ice_template = ice_template
        self.prompt_template = prompt_template
        self.labels = labels
        self.seed = seed

    def topk_search(self):
        np.random.seed(self.seed)
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]

        logger.info("Retrieving data for test set...")
        for entry in tqdm.tqdm(res_list, disable=not self.is_main_process):
            idx = entry['metadata']['id']

            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, min(self.candidate_num, len(self.index_ds)))[1][0].tolist()
            candidates = []
            mdl_scores = []
            for j in range(self.select_time):
                if j == 0:
                    rand_idx_list = near_ids[:self.ice_num]
                else:
                    rand_idx_list = np.random.choice(near_ids, self.ice_num, replace=False)
                    rand_idx_list = [int(i) for i in rand_idx_list]
                candidates.append(rand_idx_list)

                ice = self.generate_ice(rand_idx_list, ice_template=self.ice_template)
                mask_length = len(self.tokenizer(ice + self.ice_eos_token, verbose=False)['input_ids'])
                if self.labels is None:
                    labels = self.get_labels(self.ice_template, self.prompt_template)
                else:
                    labels = self.labels
                prompt_list = []
                for label in labels:
                    prompt = self.generate_label_prompt(idx, ice, label, self.ice_template, self.prompt_template)
                    prompt_list.append(prompt)
                loss_list = self.cal_ce(prompt_list, mask_length=mask_length)
                probs = np.exp(-np.array(loss_list))
                normalized_probs = probs / probs.sum(0, keepdims=True)
                neg_entropy = -entropy(normalized_probs, label_dim=0)
                mdl_scores.append(neg_entropy)

            rtr_idx_list[idx] = candidates[mdl_scores.index(max(mdl_scores))]
            rtr_idx_list[idx] = [int(i) for i in rtr_idx_list[idx]]

        return rtr_idx_list

    def retrieve(self):
        return self.topk_search()

    def cal_ce(self, input_texts: List[str], mask_length=None):
        if self.metric_model is None:
            logger.info(f'Load model {self.metric_model} for calculating MDL...')
            self.metric_model = AutoModelForCausalLM.from_pretrained(self.ce_model_name)
            self.metric_model.to(self.device)
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        outputs = self.metric_model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        shift_logits = shift_logits.view(-1, shift_logits.size(-1))
        loss = loss_fct(shift_logits, shift_labels.view(-1)).view(shift_labels.size())
        if mask_length is not None:
            mask = torch.cat([torch.zeros([loss.shape[0], mask_length], dtype=torch.float),
                              torch.ones([loss.shape[0], loss.shape[-1] - mask_length], dtype=torch.float)], -1)
            mask = mask.to(self.device)
            loss = torch.mul(mask, loss)

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        if mask_length is not None:
            lens -= mask_length
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
