# here we do the testing. haha
# added more codes in the main branch. 

from datasets import load_dataset
from datasets import Dataset
from openicl import DatasetReader

'''
def gen(shards):
    for shard in shards:
        with open(shard) as f:
            for line in f:
                yield {"line": line}
shards = [f"data{i}.txt" for i in range(32)]
ds = Dataset.from_generator(gen, gen_kwargs={"shards": shards})
'''


# Loading dataset from huggingface
# lets do sst5
dataset = load_dataset('SetFit/sst5')

# Define a DatasetReader, with specified column names where input and output are stored.
data = DatasetReader(dataset, input_columns=['text'], output_column='label')

from openicl import PromptTemplate
tp_dict = {
    0: "</E>Very Negative Movie Review: </text>",
    1: "</E>Negative Movie Review: </text>",
    2: "</E>Neutral Movie Review: </text>" ,
    3: "</E>Positive Movie Review: </text>" ,
    4: "</E>Very Positive Movie Review: </text>" 
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