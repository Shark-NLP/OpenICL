"""PPL Inferencer"""

import json
import torch
from openicl import PromptTemplate
from openicl.icl_retriever import *
from openicl.icl_evaluator import *
from openicl.icl_inferencer.icl_base_inferencer import BaseInferencer, PPLInferencerOutputHandler
from openicl.icl_inferencer.icl_ppl_inferencer import PPLInferencer
from openicl.utils.logging import get_logger
from openicl.utils.api_service import *
from typing import List, Union, Optional
from tqdm import tqdm
from tqdm import trange
from transformers import PretrainedConfig
from accelerate import Accelerator

logger = get_logger(__name__)


class ChannelInferencer(PPLInferencer):
    """PPL In-context Learning Inferencer Class
        Channel In-context Learning Inferencer.
        We recommend you to use ppl inferencer instead of channel inferencer

    """

    def inference(self, retriever: BaseRetriever, ice_template: Optional[PromptTemplate] = None,
                  prompt_template: Optional[PromptTemplate] = None, output_json_filepath: Optional[str] = None,
                  output_json_filename: Optional[str] = None, normalizing_str: Optional[str] = None) -> List:
        # 1. Preparation for output logs
        output_handler = PPLInferencerOutputHandler(self.accelerator)

        sub_predictions = []
        ppl = []
        ice = []

        if output_json_filepath is None:
            output_json_filepath = self.output_json_filepath
        if output_json_filename is None:
            output_json_filename = self.output_json_filename

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
            index = 0
            prompt_list = []
            sub_ppl_list = []
            context_length_list = []

            # 5.1 Generate prompts of current label and truncate 
            for idx in range(len(ice_idx_list)):
                prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                         prompt_template=prompt_template,
                                                         remain_sep=True)
                if self.max_model_token_num is not None and self.api_name != 'gpt3':
                    prompt_token_num = self.get_input_token_num(prompt)
                    while len(ice_idx_list[idx]) > 0 and prompt_token_num > self.max_model_token_num:
                        ice_idx_list[idx] = ice_idx_list[idx][:-1]
                        ice[idx] = retriever.generate_ice(ice_idx_list[idx], ice_template=ice_template)
                        prompt = retriever.generate_label_prompt(idx, ice[idx], label, ice_template=ice_template,
                                                                 prompt_template=prompt_template)
                        prompt_token_num = self.get_input_token_num(prompt)

                prompt_sep = prompt
                if prompt_template is not None:
                    sep_token = prompt_template.sep_token
                else:
                    sep_token = ice_template.sep_token
                sep_pos = prompt_sep.find(sep_token)
                context = prompt_sep[0:sep_pos]
                prompt = prompt_sep.replace(sep_token, '')
                context_length_list.append(self.get_input_token_num(context))
                prompt_list.append(prompt)

            # 5.2 Get PPL
            logger.info(f"Calculating PPL for prompts labeled '{label}'")
            for idx in trange(0, len(prompt_list), self.batch_size, disable=not self.is_main_process):
                sub_prompt_list = prompt_list[idx:idx + self.batch_size]
                sub_context_length_list = context_length_list[idx:idx + self.batch_size]

                with torch.no_grad():
                    sub_res = self.__get_ppl(input_texts=sub_prompt_list, mask_length=sub_context_length_list)
                for res, prompt in zip(sub_res, sub_prompt_list):
                    sub_ppl_list.append(res)
                    output_handler.save_prompt_and_ppl(label, prompt[len(ice[idx]):], prompt, res, index)
                    index = index + 1
            ppl.append(sub_ppl_list)

        # 6. Get lowest PPL class as predictions
        ppl = list(zip(*ppl))
        for single_ppl in ppl:
            sub_predictions.append(labels[single_ppl.index(min(single_ppl))])
        output_handler.save_predictions(sub_predictions)

        # 7. Output
        output_handler.subprocess_write_to_json(output_json_filepath, output_json_filename)
        if self.accelerator is not None:
            self.accelerator.wait_for_everyone()
        output_handler.merge_to_main_process(output_json_filepath, output_json_filename)
        output_handler.write_to_json(output_json_filepath, output_json_filename)

        return [sample['prediction'] for sample in output_handler.results_dict.values()]
