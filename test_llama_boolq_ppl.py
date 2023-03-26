from datasets import load_dataset 
from openicl import DatasetReader
from openicl import PromptTemplate, PPLInferencer, GenInferencer, ZeroRetriever, AccEvaluator

dataset=load_dataset("super_glue", "boolq")
data=DatasetReader(dataset, input_columns=['question', 'passage',], output_column='label')
template=PromptTemplate(
    {
       0: "</E></C>\n Can we know </X> based on context above? No.",
       1: "</E></C>\n Can we know </X> based on context above? Yes."
    },
    column_token_map={'passage':'</C>', 'question':'</X>'},
    ice_token='</E>' 
)
rtr = ZeroRetriever(data, test_split='validation')
infr = PPLInferencer(model_name='decapoda-research/llama-7b-hf', output_json_filename='llama_boolq_ppl')
predictions = infr.inference(rtr, ice_template=template)
score = AccEvaluator().score(predictions=predictions, references=data.dataset['validation']['label'])
print(score)