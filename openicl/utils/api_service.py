import json
import requests
import torch
import os
import openai
import time
import numpy as np

API_NAME_LIST = ['opt-175b', 'gpt3']
API_REQUEST_CONFIG = {
    'opt-175b': {
        'URL' : "http://10.140.1.159:6010/completions",
        'headers' : {
            "Content-Type": "application/json; charset=UTF-8"
        }
    },
    'gpt3': {
    }
}
PROXIES = {"https": "", "http": ""}


def is_api_available(api_name):
    if api_name == None:
        return False
    return True if api_name in API_NAME_LIST else False


def api_get_ppl(api_name, input_texts):
    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 0, "echo": True}
        response = json.loads(
                requests.post(API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload), headers=API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        lens = np.array([len(r['logprobs']['tokens']) for r in response['choices']])
        ce_loss = np.array([-sum(r['logprobs']['token_logprobs']) for r in response['choices']])
        return ce_loss / lens
        # lens = np.array([len(r['logprobs']['tokens']) for r in response['choices']])
        # loss_lens = np.array([len(r['logprobs']['token_logprobs']) for r in response['choices']])
        #
        # loss = [r['logprobs']['token_logprobs'] for r in response['choices']]
        #
        # max_len = loss_lens.max()
        # loss_pad = list(map(lambda l: l + [0] * (max_len - len(l)), loss))
        # loss = -np.array(loss_pad)
        #
        # loss = torch.tensor(loss)
        # ce_loss = loss.sum(-1).cpu().detach().numpy()  # -log(p(y))
        # return ce_loss / lens
    if api_name == 'gpt3':
        raise NotImplementedError("GPT3 API doesn't support PPL calculation")


def api_get_tokens(api_name, input_texts):
    length_list = [len(text) for text in input_texts]
    if api_name == 'opt-175b':
        pyload = {"prompt": input_texts, "max_tokens": 100, "echo": True}
        response = json.loads(
                requests.post(API_REQUEST_CONFIG[api_name]['URL'], data=json.dumps(pyload), headers=API_REQUEST_CONFIG[api_name]['headers'], proxies=PROXIES).text)
        return [r['text'] for r in response['choices']], [r['text'][length:] for r, length in zip(response['choices'], length_list)]
    if api_name == 'gpt3':
        openai.api_key = os.getenv("OPENAI_API_KEY")
        response = openai.Completion.create(
            engine="text-davinci-003",
            prompt=input_texts,
            temperature=0,
            max_tokens=256,
            top_p=1.0,
            frequency_penalty=0.0,
            presence_penalty=0.0
        )
        time.sleep(3)
        return [(input + r['text']) for r, input in zip(response['choices'], input_texts)], [r['text'] for r in response['choices']]
    