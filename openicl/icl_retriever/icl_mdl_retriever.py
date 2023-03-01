'''MDL retriever'''

from openicl import DatasetReader, PromptTemplate
from openicl.icl_retriever import TopkRetriever
from openicl.utils.calculate import entropy
from typing import List, Union, Optional, Tuple
from transformers import AutoModelForCausalLM
import tqdm
import torch
import numpy as np
from accelerate import Accelerator

class MDLRetriever(TopkRetriever):
    """MDL In-context Learning Retriever Class
        Class of MDL Retriever.
        
    Attributes:
        dataset_reader (DatasetReader): An instance of the `DatasetReader` class.
        ice_separator (str, optional): A string that separates each in-context example.
        ice_eos_token (str, optional): A string that is added to the end of in-context examples.
        prompt_eos_token (str, optional): A string that is added to the end of the prompt.
        ice_num: The number of data in the in-context examples.
        index_split (str, optional): A string for the index dataset name. The index dataset is used to select data for in-context examples. The default is 'train'.
        test_split (str, optional): A string for the generation dataset name. The test dataset is used to generate prompts for each data. The default is 'test'.
        index_ds (Dataset): The index dataset. Used to select data for in-context examples.
        test_ds (Dataset): The test dataset. Used to generate prompts for each data.
        accelerator (Accelerator, optional): An instance of the `Accelerator` class, used for multiprocessing.
        batch_size (int, optional): Batch size for the `DataLoader`. 
        model (SentenceTransformer): An instance of `SentenceTransformer` class, used to calculate embeddings.
        tokenizer (AutoTokenizer): Tokenizer for `model`.
        index (IndexIDMap): Index generated with FAISS.
    """
    metric_model = None
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
                 ce_model_name: Optional[str] = 'gpt2-xl',
                 batch_size: Optional[int] = 1,
                 select_time: Optional[int] = 5,
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        super().__init__(dataset_reader, ice_separator, ice_eos_token, prompt_eos_token, sentence_transformers_model_name, ice_num, index_split, test_split, tokenizer_name, batch_size, accelerator)
        self.ce_model_name = ce_model_name
        self.candidate_num = candidate_num
        self.select_time = select_time

        
    def topk_search(self):
        res_list = self.forward(self.dataloader)
        rtr_idx_list = [[] for _ in range(len(res_list))]
        for entry in tqdm.tqdm(res_list):
            idx = entry['metadata']['id']

            embed = np.expand_dims(entry['embed'], axis=0)
            near_ids = self.index.search(embed, min(self.candidate_num, len(self.index_ds)))[1][0].tolist()
            candidates = []
            mdl_scores = []
            for _ in range(self.select_time):
                rand_idx_list = np.random.choice(near_ids, self.ice_num, replace=False)
                candidates.append(rand_idx_list)
                input_texts = []
                for rand_idx in rand_idx_list:
                    input_texts.append(self.select_datalist[rand_idx])
                loss_list = self.cal_ce(input_texts)
                probs = np.exp(-np.array(loss_list))
                normalized_probs = probs / probs.sum(0, keepdims=True)
                neg_entropy = -entropy(normalized_probs, label_dim=0)
                mdl_scores.append(neg_entropy)
            
            rtr_idx_list[idx] = candidates[mdl_scores.index(max(mdl_scores))]
            rtr_idx_list[idx] = [int(i) for i in rtr_idx_list[idx]]
            
        return rtr_idx_list   
        
        
    def retrieve(self):
        return self.topk_search()
        
        
    def cal_ce(self, input_texts: List[List]):
        if self.metric_model is None:
            print("load metric model")
            self.metric_model = AutoModelForCausalLM.from_pretrained(self.ce_model_name)
            self.metric_model.to(self.device)
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.metric_model(**inputs)
        
        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()
        
        
        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())
        
        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
    