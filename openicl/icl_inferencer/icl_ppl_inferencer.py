'''PPL Inferencer'''

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, PPLInferencerOutputHandler
from openicl.utils.api_service import *
from typing import List, Union, Optional
from tqdm import tqdm
from tqdm import trange
from transformers import PretrainedConfig
from accelerate import Accelerator


class PPLInferencer(BaseInferencer):
    """PPL In-context Learning Inferencer Class
        Perplexity-based In-context Learning Inferencer.
        
    Attributes:
        model (AutoModelForCausalLM, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (AutoTokenizer or GPT2Tokenizer, optional): Tokenizer for `model`.
        max_model_token_num (int, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (int, optional): Batch size for the `DataLoader`. 
        accelerator (Accelerator, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (str, optional): File path for output JSON file. 
        output_json_filename (str, optional): File name for output JSON file. 
        call_api (bool, optional): If True, an API for LM models will be used.   
        labels (List, optional): A list of labels for all classes.
    """    
    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 labels: Optional[List] = None
    ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator, output_json_filepath, output_json_filename, api_name)
        self.labels = labels


    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None, prompt_template: Optional[PromptTemplate] = None) -> List:
        # 1. Preparation for output logs
        output_handler = PPLInferencerOutputHandler(self.accelerator)
        sub_predictions = []
        ppl = []
        ice = []
        
        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        
        # 3. Get labels of all the classes
        if self.labels is None:
            labels = retriever.get_labels(ice_template=ice_template, prompt_template=prompt_template)    
        else:
            labels = self.labels
        
        # 4. Generate in-context examples for testing inputs
        for idx in range(len(ice_idx_list)):
            ice.append(retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template))
        output_handler.save_ice(ice)

        # 5. Calculating PPL for prompts in each label's class
        for label in labels:
            prompt_list = []
            sub_ppl_list = []
            
            # 5.1 Generate prompts of current label and truncate 
            for idx in range(len(ice_idx_list)):
                prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template, prompt_template=prompt_template)
                if self.max_model_token_num is not None:
                    prompt_token_num = self.get_input_token_num(prompt)
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                            ice_idx_list[idx] = ice_idx_list[idx][:-1]
                            ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template)
                            prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template, prompt_template=prompt_template)
                            prompt_token_num = self.get_input_token_num(prompt)
                prompt_list.append(prompt)
            
            # 5.2 Get PPL
            print(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                with torch.no_grad():
                    sub_res = self.__get_ppl(sub_prompt_list).tolist()
                for res, prompt in zip(sub_res, sub_prompt_list):
                    sub_ppl_list.append(res)
                    output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, idx)
            ppl.append(sub_ppl_list)

        # 6. Get lowest PPL class as predictions
        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(min(single_ppl))])
        output_handler.save_predictions(sub_predictions)

        # 7. Output
        output_handler.subprocess_write_to_json(self.output_json_filepath, self.output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(self.output_json_filepath, self.output_json_filename)
        output_handler.write_to_json(self.output_json_filepath, self.output_json_filename)
        
        return [sample['prediction'] for sample in output_handler.results_dict.values()]


    def __get_ppl(self, input_texts: List[str]):
        if self.call_api:
            return api_get_ppl(self.api_name, input_texts)
        self.tokenizer.padding_side = "right"
        inputs = self.tokenizer(input_texts, padding=True, return_tensors='pt', truncation=True)
        inputs = {k: v.to(self.model.device) for k, v in inputs.items()}
        outputs = self.model(**inputs)

        shift_logits = outputs.logits[..., :-1, :].contiguous()
        shift_labels = inputs["input_ids"][..., 1:].contiguous()

        loss_fct = torch.nn.CrossEntropyLoss(reduction='none', ignore_index=self.tokenizer.pad_token_id)
        loss = loss_fct(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1)).view(
        shift_labels.size())

        lens = (inputs["input_ids"] != self.tokenizer.pad_token_id).sum(-1).cpu().numpy()
        ce_loss = loss.sum(-1).cpu().detach().numpy() / lens
        return ce_loss
