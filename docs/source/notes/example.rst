:github_url: https://github.com/Shark-NLP/OpenICL

A Simple Example 
========================

Following example shows you how to perform ICL on sentiment classification dataset. More examples and tutorials can be found at our github `repository <https://github.com/Shark-NLP/OpenICL/tree/main/examples>`_. 

Step 1: Load and prepare data
----------------------------------

.. code-block:: python

    from datasets import load_dataset
    from openicl import DatasetReader

    # Loading dataset from huggingface
    dataset = load_dataset('gpt3mix/sst2')

    # Define a DatasetReader, with specified column names where input and output are stored.
    data = DatasetReader(dataset, input_columns=['text'], output_column='label')


Step 2: Define the prompt template (Optional)
---------------------------------------------

.. code-block:: python

    from openicl import PromptTemplate
    tp_dict = {
        0: "</E>Positive Movie Review: </text>",
        1: "</E>Negative Movie Review: </text>" 
    }

    template = PromptTemplate(tp_dict, {'text': '</text>'}, ice_token='</E>')

The placeholder `</E>` and `</text>` will be replaced by in-context examples and testing input, respectively. 

Step 3: Initialize the Retriever
--------------------------------

.. code-block:: python

    from openicl import TopkRetriever
    # Define a retriever using the previous `DataLoader`.
    # `ice_num` stands for the number of data in in-context examples.
    retriever = TopkRetriever(data, ice_num=8)

Here we use the popular `TopK <https://arxiv.org/abs/2101.06804>`_ method to build the retriever.

Step 4: Initialize the Inferencer
---------------------------------

.. code-block:: python

    from openicl import PPLInferencer
    inferencer = PPLInferencer(model_name='distilgpt2')

Step 5: Inference and scoring
-----------------------------

.. code-block:: python

    from openicl import AccEvaluator
    # the inferencer requires retriever to collect in-context examples, as well as a template to wrap up these examples.
    predictions = inferencer.inference(retriever, ice_template=template)
    # compute accuracy for the prediction
    score = AccEvaluator().score(predictions=predictions, references=data.references)
    print(score)
    