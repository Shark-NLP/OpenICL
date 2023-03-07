<div align="center">
<img src="https://s1.ax1x.com/2023/03/07/ppZfEmq.png" border="0" width=600px/>
</div>

------
![version](https://img.shields.io/badge/version-0.1.0-blue)

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="https://arxiv.org/abs/2303.02913">Paper</a> •
  <a href="#docs">Docs</a> •
  <a href="#citation">Citation</a> 
</p>

## Overview
OpenICL provides an easy interface for in-context learning, with many state-of-the-art retrieval and inference methods built in to facilitate systematic comparison of LMs and fast research prototyping. Users can easily incorporate different retrieval and inference methods, as well as different prompt instructions into their workflow. 
<div align="center">
<img src="https://s1.ax1x.com/2023/03/07/ppZWjmt.jpg"  border="0" />
</div>

## Installation
**Note: Please use Python 3.7+ for OpenICL**

### Using Pip
```
pip install openicl
```

### Using Git
Clone the repository from github:
```
git clone https://github.com/Shark-NLP/OpenICL
cd OpenICL
pip install -e .
```

## Quick Start
This example shows you how to perform ICL on sentiment classification dataset. 

#### Step 1: Load and prepare data
```python
from datasets import load_dataset
from openicl import DatasetReader

# Loading dataset from huggingface
dataset = load_dataset('gpt3mix/sst2')

# Define a DatasetReader, with specified column names where input and output are stored.
data = DatasetReader(dataset, input_columns=['text'], output_column='label')
```

#### Step 2: Define the prompt template (Optional)
```python
from openicl import PromptTemplate
tp_dict = {
    0: "</E>Positive Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>" 
}

template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
```
The placeholder `</E>` and `</text>` will be replaced by in-contedt examples and testing input, respectively. 

#### Step 3: Initialize the Retriever
```python
from openicl import TopkRetriever
# Define a retriever using the previous `DataLoader`.
# `ice_num` stands for the number of data in in-context examples.
retriever = TopkRetriever(data, ice_num=8)
```
Here we use the popular <a href="https://arxiv.org/abs/2101.06804">TopK</a> method to build the retriever. 

#### Step 4: Initialize the Inferencer 
```python
from openicl import PPLInferencer
inferencer = PPLInferencer(model_name='distilgpt2')
```

#### Step 5: Inference and scoring
```python
from openicl import AccEvaluator
# In Step 3, we set `ice_token` for `template`.
# Therefore, `template` serves as both `ice_template` and `prompt_template`
predictions = inferencer.inference(retriever, ice_template=template)

score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)
```



## Docs
coming soon...

## Citation
If you find this repository helpful, feel free to cite our paper:
```bibtex
@article{wu2023openicl,
  title={OpenICL: An Open-Source Framework for In-context Learning},
  author={Zhenyu Wu, Yaoxiang Wang, Jiacheng Ye, Jiangtao Feng, Jingjing Xu, Yu Qiao, Zhiyong Wu},
  journal={arXiv preprint arXiv:2303.02913},
  year={2023}
}
```