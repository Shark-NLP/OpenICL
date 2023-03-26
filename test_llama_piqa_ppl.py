from datasets import load_dataset 
from openicl import DatasetReader
from openicl import PromptTemplate, PPLInferencer, GenInferencer, ZeroRetriever, AccEvaluator

dataset=load_dataset('piqa')
data=DatasetReader(dataset, input_columns=['goal', 'sol1', 'sol2'], output_column='label')
template=PromptTemplate(
    {
       0: "</E>Question: </G>\nAnswer: </S1>",
       1: "</E>Question: </G>\nAnswer: </S2>"
    },
    column_token_map={'sol1':'</S1>', 'sol2':'</S2>', 'goal':"</G>"},
    ice_token='</E>' 
)
rtr = ZeroRetriever(data, test_split='validation')
infr = PPLInferencer(model_name='decapoda-research/llama-7b-hf', output_json_filename='llama_piqa_ppl')
predictions = infr.inference(rtr, ice_template=template)
score = AccEvaluator().score(predictions=predictions, references=data.dataset['validation']['label'])
print(score)
