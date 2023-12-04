# CS532-Project

## Pyspark Inference Latency with Sentiment Analysis Model
Authors: Lincy Pattanaik, Anirudh Hariharan, Amizhthan Madan and Yueyang Yu

### Project structure:
- streaming.py: processes tweets in batches specified by 'maxFilesPerTrigger' parameter on line 36. 

- /data
    - /data_raw: contains the data after preprocssing, used for model developement
    - /data_test - used for model validation and generating streams for real time pipeline 
    - /output_folder - batched files for perfomance (latency) evaluation

- /docs - files submitted for grading, contains proposal, milestone and final presentation file.

- /model_development
    - Preprocessing.py: text cleaing and generating labels for tweets
    - 'LR model dev'.ipynb: python notebook for model tuning
        - long runtime for gridsearch cv 
    - model_lr.py : saving final model in /model

- /model 
    - saved model from model_development used in streaming.py and /tests/evaluation.py

- /tests
    - evaluation.py: latency measurement (total, preprpcessing, model inference) and plotting


### To run the final code
- python streaming.py
    - processes data in 5 batches from 5 csv files in /data/data_test folder, classifies the tweet and prints sentiment category and latency on the console
