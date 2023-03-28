"""chain-of-thought inferencer"""

import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, GenInferencerOutputHandler
from typing import List, Union, Optional
from tqdm import tqdm
from transformers import PretrainedConfig
from openicl.utils.api_service import *
from openicl.utils.icl_common_utils import get_dataloader, get_generation_prompt_list_from_retriever_indices
from openicl.utils.logging import get_logger
from accelerate import Accelerator

logger = get_logger(__name__)


class CoTInferencer(BaseInferencer):
    """COT In-context Learning Inferencer Class
        Chain-of-Thought In-context Learning Inferencer.
        
    Attributes:
        model (:obj:`AutoModelForCausalLM`, optional): Local PLM (loaded from Hugging Face), which can be initialized by name or a config class. 
        tokenizer (:obj:`AutoTokenizer` or :obj:`GPT2Tokenizer`, optional): Tokenizer for :obj:`model`.
        max_model_token_num (:obj:`int`, optional): Maximum number of tokenized words allowed by the LM. 
        batch_size (:obj:`int`, optional): Batch size for the :obj:`DataLoader`. 
        accelerator (:obj:`Accelerator`, optional): An instance of the `Accelerator` class, used for multiprocessing.
        output_json_filepath (:obj:`str`, optional): File path for output `JSON` file. 
        output_json_filename (:obj:`str`, optional): File name for output `JSON` file. 
        api_name (:obj:`str`, optional): Name of API service. 
        call_api (:obj:`bool`): If ``True``, an API for LM models will be used, determined by :obj:`api_name`. 
        gen_field_replace_token (:obj:`str`, optional): Used to replace the generation field token when generating prompts.
        generation_kwargs (:obj:`Dict`, optional): Parameters for the :obj:`model.generate()` method. 
        cot_list (:obj:`list`, optional): A list of sentences used for multiple-step generations.
    """

    def __init__(self,
                 cot_list: Optional[List[str]] = [],
                 model_name: Optional[str] = 'gpt2-xl',
                 tokenizer_name: Optional[str] = None,
                 max_model_token_num: Optional[int] = None,
                 model_config: Optional[PretrainedConfig] = None,
                 batch_size: Optional[int] = 1,
                 gen_field_replace_token: Optional[str] = '',
                 generation_kwargs={"max_new_tokens": 100},
                 accelerator: Optional[Accelerator] = None,
                 output_json_filepath: Optional[str] = "./icl_inference_output",
                 output_json_filename: Optional[str] = "predictions",
                 api_name: Optional[str] = None,
                 model_parallel: Optional[bool] = False,
                 **kwargs
                 ) -> None:
        super().__init__(model_name, tokenizer_name, max_model_token_num, model_config, batch_size, accelerator,
                         output_json_filepath, output_json_filename, api_name, model_parallel, **kwargs)
        self.cot_list = cot_list
        self.gen_field_replace_token = gen_field_replace_token
        self.generation_kwargs = generation_kwargs

    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        num = len(retriever.test_ds)
        output_handler = GenInferencerOutputHandler(num, self.accelerator)
        index = 0
        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

        # 2. Get results of retrieval process
        ice_idx_list = retriever.retrieve()
        cot_list_len = len(self.cot_list)

        # 3. Generate prompts for testing input 
        prompt_list = get_generation_prompt_list_from_retriever_indices(ice_idx_list, retriever, self.tokenizer,
                                                                        self.gen_field_replace_token,
                                                                        max_model_token_num=self.max_model_token_num,
                                                                        ice_template=ice_template,
                                                                        prompt_template=prompt_template)
        if cot_list_len > 0:
            prompt_list = [prompt + self.cot_list[0] for prompt in prompt_list]

        # 4. Inference for `max((len(self.cot_list) + 1), 1)` times
        for idx in range(0, max(cot_list_len, 1)):
            index = 0
            cot_idx = idx + 1
            # 4-1. Wrap prompts with Dataloader
            dataloader = get_dataloader(prompt_list, self.batch_size)
            output_handler.save_orgin_prompts(prompt_list)

            for entry in tqdm(dataloader, disable=not self.is_main_process):
                # 4-2-1. Inference with local model
                if not self.call_api:
                    with torch.no_grad():
                        tokenized_data = self.tokenizer.batch_encode_plus(entry, padding=True, return_tensors='pt').to(
                            self.device)
                        prompt_len = int(tokenized_data.attention_mask.shape[1])
                        if 't5' in self.model_name:
                            prompt_len = 0
                        outputs = self.model.generate(input_ids=tokenized_data.input_ids,
                                                      attention_mask=tokenized_data.attention_mask,
                                                      eos_token_id=self.tokenizer.eos_token_id,
                                                      pad_token_id=self.tokenizer.pad_token_id,
                                                      **self.generation_kwargs)
                        outputs = outputs.tolist()
                        complete_output = self.tokenizer.batch_decode(outputs[:], skip_special_tokens=True)
                        generated = self.tokenizer.batch_decode([output[prompt_len:] for output in outputs],
                                                                skip_special_tokens=True)
                # 4-2-2. Inference with remote API
                else:
                    complete_output, generated = api_get_tokens(self.api_name, entry)

                # 4-2-3. Save current output
                for prediction, output in zip(generated, complete_output):
                    if 't5' in self.model_name:
                        output = prompt_list[index] + output
                    output_handler.save_prediction_and_output(prediction, output, index)
                    prompt_list[index] = output
                    index = index + 1

            # 4-3. Output for current step
            if cot_idx < cot_list_len:
                filename = output_json_filename + f'_step{idx}'
            else:
                filename = output_json_filename
            output_handler.subprocess_write_to_json(output_json_filepath, filename)
            if self.accelerator is not None:
                self.accelerator.wait_for_everyone()
            output_handler.merge_to_main_process(output_json_filepath, filename)
            output_handler.write_to_json(output_json_filepath, filename)

            # 4-4. Check for next string in `self.cot_list`
            if cot_idx < cot_list_len:
                prompt_list = [(prompt + str(self.cot_list[cot_idx])) for prompt in prompt_list]
            else:
                break
        return [sample['prediction'] for sample in output_handler.results_dict.values()]
