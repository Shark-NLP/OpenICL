"""Direct Generation Inferencer"""

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from openicl.utils.api_service import * 
from openicl.utils.icl_common_utils import get_dataloader, get_generation_prompt_list_from_retriever_indices
from openicl.utils.logging import get_logger, SUBPROCESS_LOG_LEVEL
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import PretrainedConfig
from accelerate import Accelerator

logger = get_logger(__name__)

class GenInferencer(BaseInferencer):
    """Generation In-context Learning Inferencer Class
        In-context Learning Inferencer for Directly Generation.
    Attributes:
        model (AutoModelForCausalLM, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (AutoTokenizer or GPT2Tokenizer, optional): Tokenizer for `model`.
        max_model_token_num (int, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (int, optional): Batch size for the `DataLoader`. 
        accelerator (Accelerator, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (str, optional): File path for output JSON file. 
        output_json_filename (str, optional): File name for output JSON file. 
        call_api (bool, optional): If True, an API for LM models will be used.   
        gen_field_replace_token (str, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (Dict, optional): Parameters for the `model.generate()` method. 
    """
    def __init__(self,
                 model_name: Optional[str] = 'gpt2-xl', 
                 tokenizer_name: Optional[str] = None, 
                 max_model_token_num: Optional[int] = None, 
                 model_config: Optional[PretrainedConfig] = None, 
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs = {"max_new_tokens": 100,
                                      "do_sample": False},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 **kwargs
    ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator, output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        if not self.is_main_process:
            logger.setLevel(SUBPROCESS_LOG_LEVEL)
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs
        
    
    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None, prompt_template: Optional[PromptTemplate] = None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0
        
        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        
        # 3. Generate prompts for testing input 
        prompt_list = get_generation_prompt_list_from_retriever_indices(ice_idx_list, retriever, self.tokenizer, self.gen_field_replace_token, max_model_token_num=self.max_model_token_num, ice_template=ice_template, prompt_template=prompt_template)
        output_handler.save_orgin_prompts(prompt_list)
        
        # 4. Wrap prompts with Dataloader
        dataloader = get_dataloader(prompt_list, self.batch_size)
        
        # 5. Inference for prompts in each batch 
        logger.info("Starting inference process...")
        for entry in tqdm(dataloader, disable=not self.is_main_process):
            # 5-1. Inference with local model
            if not self.call_api:
                with torch.no_grad():
                    tokenized_data = self.tokenizer.batch_encode_plus(entry, padding=True, return_tensors='pt').to(self.device)
                    prompt_len = int(tokenized_data.attention_mask.shape[1])
                    outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                attention_mask=tokenized_data.attention_mask,
                                                eos_token_id=self.tokenizer.eos_token_id,
                                                pad_token_id=self.tokenizer.pad_token_id,
                                                **self.generation_kwargs)
                    outputs = outputs.tolist()
                    complete_output = self.tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
                    generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs], skip_special_tokens=True)
            # 5-2. Inference with remote API
            else:
                complete_output, generated = api_get_tokens(self.api_name, entry)
            
            # 5-3. Save current output
            for prediction, output in zip(generated, complete_output): 
                output_handler.save_prediction_and_output(prediction, output, index)
                index = index + 1
        
        # 6. Output 
        output_handler.subprocess_write_to_json(self.output_json_filepath, self.output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(self.output_json_filepath, self.output_json_filename)
        output_handler.write_to_json(self.output_json_filepath, self.output_json_filename)
        return list(output_handler.prediction_dict.values())
