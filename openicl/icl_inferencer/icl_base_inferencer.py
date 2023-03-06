'''Basic Inferencer'''

import os
import torch
from openicl import BaseRetriever, PromptTemplate
from openicl.utils.api_service import *
from openicl.icl_evaluator import *
from transformers import AutoTokenizer, AutoModelForCausalLM, PretrainedConfig, GPT2Tokenizer, AutoConfig
from typing import List, Union, Optional
from accelerate import Accelerator
from accelerate import init_empty_weights, infer_auto_device_map

class BaseInferencer:
    """Basic In-context Learning Inferencer Class
        Base class of In-context Learning Inferencer, with no inference method.

    Attributes:
        model (AutoModelForCausalLM, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (AutoTokenizer or GPT2Tokenizer, optional): Tokenizer for `model`.
        max_model_token_num (int, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (int, optional): Batch size for the `DataLoader`. 
        accelerator (Accelerator, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (str, optional): File path for output JSON file. 
        output_json_filename (str, optional): File name for output JSON file. 
        call_api (bool, optional): If True, an API for LM models will be used.   
    """
    model = None
    tokenizer = None
    call_api = False
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
                 model_parallel: Optional[bool] = False,
                 **kwargs
    ) -> None:
        self.model_name = model_name
        self.tokenizer_name = tokenizer_name if tokenizer_name is not None else model_name
        self.accelerator = accelerator
        self.api_name = api_name
        
        if 'no_split_module_classes' not in kwargs.keys():
            kwargs['no_spilt_module_classes'] = []
        if 'device_map' not in kwargs.keys():
            kwargs['device_map'] = None
            
        no_split_module_classes = kwargs['no_spilt_module_classes']
        device_map = kwargs['device_map']

        self.__init_api()
        if not self.call_api:
            self.__init_model(self.model_name, model_config, model_parallel, device_map, no_split_module_classes)
            self.__init_tokenizer(self.tokenizer_name)
            
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        if self.model is not None:
            self.model.to(self.device)
        self.max_model_token_num = max_model_token_num
        self.batch_size = batch_size
        self.output_json_filepath = output_json_filepath
        self.output_json_filename = output_json_filename
        if not os.path.exists(self.output_json_filepath):
            os.makedirs(self.output_json_filepath)
            
    
    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None, prompt_template: Optional[PromptTemplate] = None, references: Optional[List] = None) -> List:
        raise NotImplementedError("Method hasn't been implemented yet")
    
    
    def __init_model(self, model_name, model_config, model_parallel, device_map, no_split_module_classes):
        if not model_parallel:
            if model_config is not None:
                self.model = AutoModelForCausalLM.from_config(model_config)
            else:
                self.model = AutoModelForCausalLM.from_pretrained(model_name)
        else:
            if model_config is None:
                model_config = AutoConfig.from_pretrained(model_name)
            with init_empty_weights():
                empty_model = AutoModelForCausalLM.from_config(model_config)
            
            if device_map is None:
                device_map = infer_auto_device_map(empty_model, no_split_module_classes=no_split_module_classes)
            
            self.model = AutoModelForCausalLM.from_pretrained(model_name, device_map=device_map, offload_folder="offload", offload_state_dict=True, torch_dtype=torch.float16)
            

    def __init_tokenizer(self, tokenizer_name):
        if self.api_name == 'opt-175b':
            self.tokenizer = GPT2Tokenizer.from_pretrained("facebook/opt-30b", use_fast=False)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
        self.tokenizer.padding_side = "left"
        
    
    def __init_api(self):
        if self.api_name == None:
            return
        self.call_api = is_api_available(self.api_name)
        if not self.call_api:
            UserWarning(f"api_name '{self.api_name}' is not available, Please check it")
    
        
    def get_input_token_num(self, inputs):
        return len(self.tokenizer(inputs, verbose=False)['input_ids'])    
    
    
class GenInferencerOutputHandler:
    origin_prompt_dict= {}
    output_dict = {}
    prediction_dict = {}
    results_dict = {}
    def __init__(self, 
                 num: int,
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        self.num = num
        self.accelerator = accelerator
                
    
    def subprocess_write_to_json(self, output_json_filepath: str, output_json_filename: str):
        self.results_dict = {
            str(idx): {
                'origin_prompt': self.origin_prompt_dict[str(idx)],
                'output': self.output_dict[str(idx)],
                'prediction': self.prediction_dict[str(idx)]
            } for idx in range(self.num)
        }
        if self.accelerator is not None:
            with open(f'{output_json_filepath}/process{self.accelerator.process_index}_{output_json_filename}.json', 'w', encoding='utf-8') as json_file:
                json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)


    def write_to_json(self, output_json_filepath: str, output_json_filename: str):
        with open(f'{output_json_filepath}/{output_json_filename}.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)
            
            
    def merge_to_main_process(self, output_json_filepath: str, output_json_filename: str):
        if self.accelerator is not None and self.accelerator.is_main_process:
            for pid in range(self.accelerator.num_processes):
                with open(f'{output_json_filepath}/process{pid}_{output_json_filename}.json', 'r', encoding='utf-8') as json_file:
                    subprocess_results_dict = json.load(json_file)
                    self.results_dict.update(subprocess_results_dict)
    
    
    def save_orgin_prompts(self, origin_prompts: List[str]):
        for idx, origin_prompt in enumerate(origin_prompts):
            if self.accelerator is not None:
                idx = idx * self.accelerator.num_processes + self.accelerator.process_index
            self.origin_prompt_dict[str(idx)] = origin_prompt
                
    
    def save_prediction_and_output(self, prediction, output, idx):
        if self.accelerator is not None:
            idx = idx * self.accelerator.num_processes + self.accelerator.process_index
        self.prediction_dict[str(idx)] = prediction
        self.output_dict[str(idx)] = output
        

class PPLInferencerOutputHandler:
    results_dict = {}
    def __init__(self, 
                 accelerator: Optional[Accelerator] = None
    ) -> None:
        self.accelerator = accelerator
    
    
    def subprocess_write_to_json(self, output_json_filepath: str, output_json_filename: str):
        if self.accelerator is not None:
            with open(f'{output_json_filepath}/process{self.accelerator.process_index}_{output_json_filename}.json', 'w', encoding='utf-8') as json_file:
                json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)
    
    
    def write_to_json(self, output_json_filepath: str, output_json_filename: str):
        with open(f'{output_json_filepath}/{output_json_filename}.json', 'w', encoding='utf-8') as json_file:
            json.dump(self.results_dict, json_file, indent=4, ensure_ascii=False)


    def merge_to_main_process(self, output_json_filepath: str, output_json_filename: str):
        if self.accelerator is not None and self.accelerator.is_main_process:
            for pid in range(self.accelerator.num_processes):
                with open(f'{output_json_filepath}/process{pid}_{output_json_filename}.json', 'r', encoding='utf-8') as json_file:
                    subprocess_results_dict = json.load(json_file)
                    self.results_dict.update(subprocess_results_dict)
                    
                    
    def save_ice(self, ice):
        for idx, example in enumerate(ice):
            if self.accelerator is not None:
                idx = idx * self.accelerator.num_processes + self.accelerator.process_index
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['in-context examples'] = example
    
    
    def save_predictions(self, predictions):
        for idx, prediction in enumerate(predictions):
            if self.accelerator is not None:
                idx = idx * self.accelerator.num_processes + self.accelerator.process_index
            if str(idx) not in self.results_dict.keys():
                self.results_dict[str(idx)] = {}
            self.results_dict[str(idx)]['prediction'] = prediction
    
    
    def save_prompt_and_ppl(self, label, input, prompt, ppl, idx):
        if self.accelerator is not None:
            idx = idx * self.accelerator.num_processes + self.accelerator.process_index
        if str(idx) not in self.results_dict.keys():
            self.results_dict[str(idx)] = {}
        if 'label: ' + str(label) not in self.results_dict[str(idx)].keys():
            self.results_dict[str(idx)]['label: ' + str(label)] = {}
        self.results_dict[str(idx)]['label: ' + str(label)]['testing input'] = input
        self.results_dict[str(idx)]['label: ' + str(label)]['prompt'] = prompt
        self.results_dict[str(idx)]['label: ' + str(label)]['PPL'] = ppl
    