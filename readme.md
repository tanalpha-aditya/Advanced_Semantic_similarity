# Project: Semantic Textual Similarity (STS) 

## About
This repository consists of STS score generation and various experiments conducted to support mono-lingual and cross-lingual embeddings.
The quality of the sentence embeddings is validated against validation data by assessing the MSE score and detailed report is present in report.pdf

## Project files
<li>run_similarity_score_en.py consists of main driver code to generate STS scores for mono lingual sentences </li>
<li>Data folder has the raw data, cleaned data and pretrained model files</li>
<li>Experiments folder have reproducible training source code for various approaches experimented:
<ul>
    <li>Word2Vec-STS.ipynb: Generate Word2vec embeddings for given corpus and average the embeddings to create sentence embeddings and apply following approaches for STS scores generation</li>
    <li>BILSTM_Sentence_Encoder.ipynb: Sentence Embeddings with BiLSTM-The cleaned sentences are padded and applied through encoder decoder model to generate sentence embeddings of equal size. The embeddings are optimized to fit MLP based regression model</li>
    <li>Do2Vec.ipynb: Doc2Vec model is trained to generate representative embeddings of sentences</li>
    <li>BERT.ipynb: SOTA The pretrained SBERT embeddings which are fine tuned on STS dataset are used for generating sentence embeddings</li>
</ul>
</li>
<li>sts_explainability.ipynb: This module helps in explaining which words or set of words are playing major role in the STS score for a given sentence pair through SHAP values</li>

## Execution
When the `run_similarity_score.py` file is run, it uses the command-line arguments . It prints a prompt asking for two sentences and then prints the predicted STS score for the sentence pair.

Make sure you are in the src directory after unzipping
```
> python run_similarity_score_en.py
Enter your first sentence:      Where is Swaroop going
Enter your second sentence:     Swaroop is leaving to college now
Generating the similarity score......
The semantic similarity score is 4.338752180337906
The explainability can be generated through sts_explainability module 
```

##Dependencies
create virtual environment of Python 3.9 version and install dependencies from requirements.txt to reproduce

##Contact
Contact author for any queries to reproduce the results

##References:
<li>Hugging face datasets</li>
<li>Efficient Estimation of Word Representation in vector space</li>
<li>https://github.com/yg211/explainable-metrics</li>



