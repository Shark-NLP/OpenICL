# here we do the testing. haha
# added more codes in the main branch. 

from datasets import load_dataset
from openicl import DatasetReader

# Loading dataset from huggingface
dataset = load_dataset('gpt3mix/sst2')

# Define a DatasetReader, with specified column names where input and output are stored.
data = DatasetReader(dataset, input_columns=['text'], output_column='label')

from openicl import PromptTemplate
tp_dict = {
    0: "</E>Positive Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>" 
}

template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

from openicl import TopkRetriever
# Define a retriever using the previous `DataLoader`.
# `ice_num` stands for the number of data in in-context examples.
retriever = TopkRetriever(data, ice_num=8)

from openicl import PPLInferencer
inferencer = PPLInferencer(model_name='distilgpt2')

from openicl import AccEvaluator
# the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
predictions = inferencer.inference(retriever, ice_template=template)
# compute accuracy for the prediction
score = AccEvaluator().score(predictions=predictions, references=data.references)
print(score)