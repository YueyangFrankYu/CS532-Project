# CS532-Project

## Pyspark Inference Latency with Sentiment Analysis Model

### Current Milestones Completed:
- Data PreProcessing
    - Remove symbols/URLs
    - Remove stopwords
    - Lower cast/ lemmatization
    - Tokenization
    - Sentiment Labeling
- Model Buiding - Logistic Regression
    - Constructing Pipeline
        - tokenizer
        - countVectorizer
        - IDF
        - Evaluator
    - Gridsearch with CV
    - Train with best params
    - Evalutaions 

### To Do
- Data Streaming
    - Try tweetpy or use another dataset 
    - Preprocess data
    - Offline 
        - Analyze performance with varying batch size
    - Online 
        - Analyze inference latency with varying stream size

- Graphs
    - Latency vs batch size, accuracy over time
    - Memory Profile

- Optional
    - Compare with some other model