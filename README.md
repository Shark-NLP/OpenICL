<div align="center">
<img src="docs/openicl.png" width=600px>
</div>


![version](https://img.shields.io/badge/version-0.1.0-blue)

------

<p align="center">
  <a href="#overview">Overview</a> •
  <a href="#installation">Installation</a> •
  <a href="#supported-methods">Methods</a> •
  <a href="#docs">Docs</a> •
  <a href="#citation">Citation</a> 
</p>

## Overview
In recent years, the rise of large language models (LLMs) has drawn attention to in-context learning (ICL), a new paradigm that leverages pre-trained language models to perform new tasks without any gradient-based training.  OpenICL provides many state-of-the-art retrieval and inference methods built in to facilitate systematic comparison and fast research prototyping. Users can easily incorporate different retrieval and inference methods, as well as different prompt instructions.
<div align="center">
<img src="docs/overview.jpg">
</div>

## Installation
**Note: Please use Python 3.7+ for OpenICL**

### Using Pip
```
pip install -e .
```
we will support  `pip install openicl`  later

### Using Git
Clone the repository from github:
```
git clone https://github.com/Shark-NLP/OpenICL
cd OpenICL
pip install -e .
```

## Supported Methods
In this library, we support the following methods for both Retrieval and Inference. (**updating...**)

### Retrieval
+ Random
+ BM25
+ TopK
+ VoteK
+ MDL

### Inference
+ Direct
+ PPL
+ CoT

## Introduction by Two Examples
### Text Classification Example
As an example of using the TopK retrieval method to evaluate the SST-2 dataset, the process is as follows:

#### Step 1: Load Dataset
```python
from datasets import load_dataset
# Loading dataset from huggingface
dataset = load_dataset('gpt3mix/sst2')
```

#### Step 2: Define a DatasetReader
```python
from openicl import DatasetReader
# Define a DatasetReader, with dataset and specified input and output columns.
data = DatasetReader(dataset, input_columns=['text'], output_column='label')
```

#### Step 3: Define a PromptTemplate (Optional)
In this example, we use a template to build both in-context examples and prompts that will be fed into the PLM. The `ice_token` placeholder could be automatically ignored when generating in-context examples in the `inference()` method.
```python
from openicl import PromptTemplate
tp_dict = {
    0: "</E>Positive Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>" 
}

# Prompt is built with in-context examples and corresponding testing input.
# `ice_token` placeholder determines the position of in-context examples.
template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')
```

#### Step 4: Define a Retriever
```python
from openicl import TopkRetriever
# Define a retriever using the previous `DataLoader`.
# `ice_num` stands for the number of data in in-context examples.
retriever = TopkRetriever(data, ice_num=8)
```

#### Step 5: Define an Inferencer 
```python
from openicl import PPLInferencer
inferencer = PPLInferencer(model_name='distilgpt2')
```

#### Step 6: Inference and Score
```python
from openicl import AccEvaluator
# In Step 3, we set `ice_token` for `template`.
# Therefore, `template` serves as both `ice_template` and `prompt_template`
predictions = inferencer.inference(retriever, ice_template=template)

score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)
```

### Semantic Parsing Example
Using OpenICL for NLG tasks has many similarities with the NLU example previously given, but there are still some differences.
+ The choice of `Inferencer` is different
+ For NLG tasks, a string-style prompt template is more appropriate as the prediction (generated) results are diverse.

Here is an example of using OpenICL to slove semantic parsing task:
```python
from openicl import DatasetReader, PromptTemplate
from openicl import BM25Retriever, GenInferencer, AccEvaluator
from datasets import load_dataset

def test_bm25_mtop():
    # 1. Loading dataset.
    dataset = load_dataset("iohadrubin/mtop")

    # 2. Define a `DatasetReader`.
    data = DatasetReader(ds, input_columns=['question'], output_column='logical_form')  

    # 3. Define a `PromptTemplate`. (string-type template)  
    tp = PromptTemplate("</E></question>\t</logical>" 
                        {
                            'question' : '</question>', 
                            'logical_form' : '</logical>',
                        }, 
                        ice_token='</E>')

    # 4. Define a retriever.
    retriever = BM25Retriever(data, ice_num=8)

    # 5. Define an inferencer.
    inferencer = GenInferencer(model_name="EleutherAI/gpt-neo-2.7B")

    # 6. Inference.
    predictions = inferencer.inference(retriever, ice_template=template)

    # 7. Score.
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    print(score)
```

## Docs
coming soon...

## Citation
coming soon...