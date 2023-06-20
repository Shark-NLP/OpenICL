from os.path import dirname as d
from os.path import abspath
import sys 
root = d(d(abspath(__file__)))
sys.path.append(root)
from collections import Counter

import json
from openicl import DatasetReader, ZeroRetriever, PromptTemplate, TopkRetriever, GenInferencer, AccEvaluator
# import fire
import re
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset
#from train_llm.test.proxy import proxy_on


def processing_answer(str):
        str = str.split(' ')[::-1]
        flag = False
        ret = ''
        for i in range(len(str)):
            s = str[i]
            for i in range(len(s)):
                if s[i].isdigit():
                    flag = True
                    ret = s
                    break
            if flag:
                break
        ret1 = ''
        for i in range(len(ret)):
            if ret[i].isdigit():
                ret1 += ret[i]
        return ret1


def main(model_path, ice_num=4, batch_size=1, max_seq_len=2048, sc_size=5):
    
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    ds = load_dataset('gsm8k', 'main', split="test[:100]")
    print(ds)
    # import pdb;pdb.set_trace()
    def processing_test(example):
        example['answer'] = example['answer'].split("#### ")[1].replace(',', '')
        return example
    
    data = DatasetReader(ds, input_columns=['question'], output_column='answer')

    ref = ds.map(processing_test)
    
    #template = PromptTemplate("</E>Q: There are 15 trees in the grove. Grove workers will plant trees in the grove today. After they are done, there will be 21 trees. How many trees did the grove workers plant today?\nA: We start with 15 trees. Later we have 21 trees. The difference must be the number of trees they planted. So, they must have planted 21 - 15 = 6 trees. The answer is 6.\n\nQ: If there are 3 cars in the parking lot and 2 more cars arrive, how many cars are in the parking lot?\nA: There are 3 cars in the parking lot already. 2 more arrive. Now there are 3 + 2 = 5 cars. The answer is 5.\n\nQ: Leah had 32 chocolates and her sister had 42. If they ate 35, how many pieces do they have left in total?\nA: Leah had 32 chocolates and Leah's sister had 42. That means there were originally 32 + 42 = 74 chocolates. 35 have been eaten. So in total they still have 74 - 35 = 39 chocolates. The answer is 39.\n\nQ: Jason had 20 lollipops. He gave Denny some lollipops. Now Jason has 12 lollipops. How many lollipops did Jason give to Denny?\nA: Jason had 20 lollipops. Since he only has 12 now, he must have given the rest to Denny. The number of lollipops he has given to Denny must have been 20 - 12 = 8 lollipops. The answer is 8.\n\nQ: Shawn has five toys. For Christmas, he got two toys each from his mom and dad. How many toys does he have now?\nA: He has 5 toys. He got 2 from mom, so after that he has 5 + 2 = 7 toys. Then he got 2 more from dad, so in total he has 7 + 2 = 9 toys. The answer is 9.\n\nQ: There were nine computers in the server room. Five more computers were installed each day, from monday to thursday. How many computers are now in the server room?\nA: There are 4 days from monday to thursday. 5 computers were added each day. That means in total 4 * 5 = 20 computers were added. There were 9 computers in the beginning, so now there are 9 + 20 = 29 computers. The answer is 29.\n\nQ: Michael had 58 golf balls. On tuesday, he lost 23 golf balls. On wednesday, he lost 2 more. How many golf balls did he have at the end of wednesday?\nA: Michael initially had 58 balls. He lost 23 on Tuesday, so after that he has 58 - 23 = 35 balls. On Wednesday he lost 2 more so now he has 35 - 2 = 33 balls. The answer is 33.\n\nQ: Olivia has $23. She bought five bagels for $3 each. How much money does she have left?\nA: She bought 5 bagels for $3 each. This means she spent 5 * $3 = $15 on the bagels. She had $23 in beginning, so now she has $23 - $15 = $8. The answer is 8.\n\nQ: </Q>\nA: </A>",
    #                      {'question':'</Q>', 'answer':'</A>'},
    #                      ice_token='</E>')
    #import pdb;pdb.set_trace()
    # prompt = open("llm_test/prompt_gsm8k_4shot.txt").readlines()
    # for _, line in enumerate(prompt):
        # if line == "Let's think step by step\n":
            # prompt[_] = "Let's think step by step\nAnswer:\n"
    # prompt = ''.join(prompt)
    prompt = """
Question: Angelo and Melanie want to plan how many hours over the next week they should study together for their test next week. They have 2 chapters of their textbook to study and 4 worksheets to memorize. They figure out that they should dedicate 3 hours to each chapter of their textbook and 1.5 hours for each worksheet. If they plan to study no more than 4 hours each day, how many days should they plan to study total over the next week if they take a 10-minute break every hour, include 3 10-minute snack breaks each day, and 30 minutes for lunch each day?
Let's think step by step
Angelo and Melanie think they should dedicate 3 hours to each of the 2 chapters, 3 hours x 2 chapters = 6 hours total.
For the worksheets they plan to dedicate 1.5 hours for each worksheet, 1.5 hours x 4 worksheets = 6 hours total.
Angelo and Melanie need to start with planning 12 hours to study, at 4 hours a day, 12 / 4 = 3 days.
However, they need to include time for breaks and lunch. Every hour they want to include a 10-minute break, so 12 total hours x 10 minutes = 120 extra minutes for breaks.
They also want to include 3 10-minute snack breaks, 3 x 10 minutes = 30 minutes.
And they want to include 30 minutes for lunch each day, so 120 minutes for breaks + 30 minutes for snack breaks + 30 minutes for lunch = 180 minutes, or 180 / 60 minutes per hour = 3 extra hours.
So Angelo and Melanie want to plan 12 hours to study + 3 hours of breaks = 15 hours total.
They want to study no more than 4 hours each day, 15 hours / 4 hours each day = 3.75
They will need to plan to study 4 days to allow for all the time they need.
The answer is 4

Question: Mark's basketball team scores 25 2 pointers, 8 3 pointers and 10 free throws.  Their opponents score double the 2 pointers but half the 3 pointers and free throws.  What's the total number of points scored by both teams added together?
Let's think step by step
Mark's team scores 25 2 pointers, meaning they scored 25*2= 50 points in 2 pointers.
His team also scores 6 3 pointers, meaning they scored 8*3= 24 points in 3 pointers
They scored 10 free throws, and free throws count as one point so they scored 10*1=10 points in free throws.
All together his team scored 50+24+10= 84 points
Mark's opponents scored double his team's number of 2 pointers, meaning they scored 50*2=100 points in 2 pointers.
His opponents scored half his team's number of 3 pointers, meaning they scored 24/2= 12 points in 3 pointers.
They also scored half Mark's team's points in free throws, meaning they scored 10/2=5 points in free throws.
All together Mark's opponents scored 100+12+5=117 points
The total score for the game is both team's scores added together, so it is 84+117=201 points
The answer is 201

Question: Bella has two times as many marbles as frisbees. She also has 20 more frisbees than deck cards. If she buys 2/5 times more of each item, what would be the total number of the items she will have if she currently has 60 marbles?
Let's think step by step
When Bella buys 2/5 times more marbles, she'll have increased the number of marbles by 2/5*60 = 24
The total number of marbles she'll have is 60+24 = 84
If Bella currently has 60 marbles, and she has two times as many marbles as frisbees, she has 60/2 = 30 frisbees.
If Bella buys 2/5 times more frisbees, she'll have 2/5*30 = 12 more frisbees.
The total number of frisbees she'll have will increase to 30+12 = 42
Bella also has 20 more frisbees than deck cards, meaning she has 30-20 = 10 deck cards
If she buys 2/5 times more deck cards, she'll have 2/5*10 = 4 more deck cards.
The total number of deck cards she'll have is 10+4 = 14
Together, Bella will have a total of 14+42+84 = 140 items
The answer is 140

Question: A group of 4 fruit baskets contains 9 apples, 15 oranges, and 14 bananas in the first three baskets and 2 less of each fruit in the fourth basket. How many fruits are there?
Let's think step by step
For the first three baskets, the number of apples and oranges in one basket is 9+15=24
In total, together with bananas, the number of fruits in one basket is 24+14=38 for the first three baskets.
Since there are three baskets each having 38 fruits, there are 3*38=114 fruits in the first three baskets.
The number of apples in the fourth basket is 9-2=7
There are also 15-2=13 oranges in the fourth basket
The combined number of oranges and apples in the fourth basket is 13+7=20
The fourth basket also contains 14-2=12 bananas.
In total, the fourth basket has 20+12=32 fruits.
The four baskets together have 32+114=146 fruits.
The answer is 146

    """

    template = PromptTemplate(f"</E>{prompt}Question: </Q>\nLet's think step by step\n</A>",
                              {'question':'</Q>', 'answer':'</A>'},
                              ice_token='</E>')

    retriever = ZeroRetriever(data)
    all_predictions = []
    
    # generation_kwargs = dict(max_new_tokens=512, do_sample=True, temperature=0.7, top_k=40)
    generation_kwargs = dict(max_new_tokens=512)
                    # {"max_gen_len": 512, "do_sample": True, "temperature": 0.8, "top_p": 0.8}
    for i in range(sc_size):
        print("**"*50)
        print("\t\t\tIteration:", str(i))
        print("**"*50)
        inferencer = GenInferencer(model_name=model, tokenizer_name=tokenizer, generation_kwargs=generation_kwargs, 
                                batch_size=batch_size, output_json_filepath=model_path.split('/')[-2], output_json_filename="gsm8k_"+str(i))
        predictions = inferencer.inference(retriever, ice_template=template)
        print(predictions[:2])
        predictions = [processing_answer(pred.split('\n\n')[0]) for pred in predictions]
        # print("**"*50)
        # print("\t\t\tProcessed prediction at iteration:", str(i))
        # print("**"*50)
        # print(predictions[:2])
        all_predictions.append(predictions)
    #import json
    # file = json.load(open("llm_llama/gsm8k.json"))
    # predictions = [file[str(i)]['prediction'] for i in range(len(file.keys()))]
    assert len(all_predictions) == sc_size
    final_prediction = []
    for i in range(len(all_predictions[0])):
        tmp_preds = []
        for j in range(sc_size):
            tmp_preds.append(all_predictions[j][i])
        counter = Counter(tmp_preds)
        if i < 5:
            print(counter)
        final_prediction.append(counter.most_common(1)[0][0])
    
    #import pdb;pdb.set_trace()
    print(final_prediction[:5], ref['answer'][:5])
    score = AccEvaluator().score(predictions=final_prediction, references=ref['answer'])
    print(score)


if __name__ == "__main__":
    # fire.Fire(main)
    
    # replace with your model_path or huggingface model name here
    main(model_path="decapoda-research/llama-7b-hf",
    ice_num=4, batch_size=8, sc_size=1)
    