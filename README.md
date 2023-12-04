# CS532-Project

## Pyspark Inference Latency with Sentiment Analysis Model
Authors: Lincy Pattanaik, Anirudh Hariharan, Amizhthan Madan and Yueyang Yu

### Project structure:
- Data
    - Data raw - contains the data after preprocssing, used by model developement
    - Data test - used by final model training
    - Output folders - batched files for latency test
- Dev process
    - Preprocessing - data prep
    - LR model dev - model tuning
        - long runtime for gridsearch cv 
    - model_lr - final model
    - Streaming - creating the streaming process
- Docs - files submitted for grading

### To run the final code just execute evaluations.py
