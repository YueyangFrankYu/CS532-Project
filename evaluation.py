from pyspark.sql import functions as F
from pyspark.sql.window import Window

import re
import time
import pyspark.sql.functions as F

import matplotlib.pyplot as plt
import os
import pandas as pd
import numpy as np

import nltk
nltk.download('punkt')
from pyspark.ml import PipelineModel
from pyspark.sql import SparkSession, DataFrame
from pyspark.sql.functions import udf, lit, col
from pyspark.sql.types import ArrayType, DoubleType, StringType, StructType, StructField, FloatType
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from pyspark.ml.feature import Tokenizer, StopWordsRemover

import re
import time
import pyspark.sql.functions as F


# Initialize Spark session
spark = SparkSession.builder.appName("CSVSplitter").getOrCreate()

# Specify the number of desired output files
num_output_files = 20

# Function to split data and save to new CSV files
def split_and_save(input_file, output_folder):
    # Read the original CSV file
    df = spark.read.csv(input_file, header=True, inferSchema=True)

    # Determine the number of rows in each split file
    total_rows = df.count()
    rows_per_file = total_rows // num_output_files

    # Add a row number to the DataFrame
    window_spec = Window.orderBy(F.monotonically_increasing_id())
    df = df.withColumn("row_num", F.row_number().over(window_spec))

    # Calculate the start and end row for each split
    split_ranges = [(i * rows_per_file + 1, (i + 1) * rows_per_file) for i in range(num_output_files - 1)]
    split_ranges.append(((num_output_files - 1) * rows_per_file + 1, total_rows))

    # Split the data and save to new CSV files
    for i, (start_row, end_row) in enumerate(split_ranges):
        split_df = df.filter((F.col("row_num") >= start_row) & (F.col("row_num") <= end_row)).drop("row_num")

        # Save the split DataFrame to a new CSV file
        output_file = f"{output_folder}/split_{i + 1}.csv"
        split_df.write.csv(output_file, header=True, mode="overwrite")

# Specify the input CSV files and output folder
input_files = ['./part-00000-29b0fbf4-8735-4a95-88cd-77da64268f21-c000.csv',
               './part-00001-29b0fbf4-8735-4a95-88cd-77da64268f21-c000.csv',
               './part-00002-29b0fbf4-8735-4a95-88cd-77da64268f21-c000.csv',
               './part-00003-29b0fbf4-8735-4a95-88cd-77da64268f21-c000.csv',
               './part-00004-29b0fbf4-8735-4a95-88cd-77da64268f21-c000.csv']
output_folder = './data/output_folder'

# Process each input file
for input_file in input_files:
    split_and_save(input_file, output_folder)

# Load model from file
model = PipelineModel.load("./model")
nltk.download('stopwords')
nltk.download('wordnet')
# Initialize Spark Session
spark = SparkSession.builder.appName("TweetStreaming").getOrCreate()

# Define the schema of data
schema = StructType([
    StructField("rawData", StringType(), True),
    StructField("tokens", StringType(), True),
    StructField("sentiment", DoubleType(), True),
    StructField("Category", StringType(), True)
])
#  Streaming from a directory
streaming_df = spark.readStream.option("delimiter", "\t").schema(schema).option("maxFilesPerTrigger", 1).csv("./data/output_folder", header=True)

def PreprocessStream(df: DataFrame):
    """
    Preprocesses the streaming data.

    :param df: DataFrame containing the data to be preprocessed.
    :return: Preprocessed DataFrame.
    """
    # Define UDFs for cleaning and lemmatization
    def clean_text(text):
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    clean_text_udf = udf(clean_text, StringType())
    lemmatize_udf = udf(lambda tokens: [WordNetLemmatizer().lemmatize(word) for word in tokens], ArrayType(StringType()))


    # Apply text cleaning
    df = df.withColumn("content", F.lower(df["rawData"])).drop("rawData").drop("tokens")
    df = df.withColumn("cleaned_content", clean_text_udf(df["content"])).drop("content")

    # Tokenization
    tokenizer = Tokenizer(inputCol="cleaned_content", outputCol="tokens")
    df = tokenizer.transform(df)
    df = df.drop("content")

    # Stop Words Removal
    stop_words = list(set(word_tokenize(' '.join(stopwords.words('english')))))
    stop_words.extend(['russian', 'u'])
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    df = remover.transform(df)

    # Lemmatization
    df = df.withColumn("words", lemmatize_udf(df["filtered_tokens"])).select("words")
    df = df.dropna()
    return df


def PredictStream(df: DataFrame):
    """
    Runs prediction on the streamed and preprocessed data.

    :param df: DataFrame containing the preprocessed data.
    :return: DataFrame with predictions.
    """

    return model.transform(df)

total=[]

def ProcessBatch(batch_df: DataFrame, batch_id: int):
    # Generate a timestamp at the start of batch processing
    batch_timestamp = time.time()

    # Perform preprocessing
    preprocessed_batch = PreprocessStream(batch_df)
    preprocesing_timestamp = time.time()

    # Perform prediction
    predictions_batch = PredictStream(preprocessed_batch)
    predictions_timestamp = time.time()

    # Add the latency
    predictions_batch = predictions_batch.withColumn("preprocessing_latency_ms", lit(round((preprocesing_timestamp- batch_timestamp) * 1000, 3)))
    predictions_batch = predictions_batch.withColumn("prediction_latency_ms", lit(round((predictions_timestamp- preprocesing_timestamp) * 1000, 3)))
    predictions_batch = predictions_batch.withColumn("total_latency_ms", lit(round((predictions_timestamp- batch_timestamp) * 1000, 3)))

    # Write to console
    predictions_batch.select("words", "features", "prediction", "preprocessing_latency_ms", "prediction_latency_ms", "total_latency_ms") \
            .write.format("console").save()
    total.append(round((predictions_timestamp- batch_timestamp) * 1000, 3))
    print(total)
    avg=np.mean(total)
    print(avg)

query = streaming_df.writeStream.foreachBatch(ProcessBatch).start()
query.awaitTermination(600) # Waits for 10 minutes
query.stop()



def plot_cost_history(cost_history):
    plt.figure()
    plt.plot([2000,4000,8000,10000,20000,40000], cost_history)
    plt.xlabel("batch size")
    plt.ylabel('total latency(in ms)')
    plt.show()

cost_history=[594.17995,637.7453,694.7976,685.3145,1055.2205,1813.464]
plot_cost_history(cost_history)

# Load the original CSV file
original_data = pd.read_csv('./data/data_test/part-00000-29b0fbf4-8735-4a95-88cd-77da64268f21-c000.csv',error_bad_lines=False)  # Replace with the actual path

# Randomly select 20k data points
selected_data = original_data.sample(n=10000, random_state=42)  # Adjust random_state for reproducibility

# Split the selected data into 20 chunks
chunks = np.array_split(selected_data, 20)

# Define the output directory- file format:output_folder_batchsize_k
output_directory = './data/output_folder_batchsize_500'

# Create the output directory if it doesn't exist
os.makedirs(output_directory, exist_ok=True)

# Save each chunk into a separate CSV file in the output directory
for i, chunk in enumerate(chunks):
    output_path = os.path.join(output_directory, f'output_chunk_{i + 1}.csv')
    chunk.to_csv(output_path, index=False)




# Load model from file
model = PipelineModel.load("./model")
nltk.download('stopwords')
nltk.download('wordnet')
# Initialize Spark Session
spark = SparkSession.builder.appName("TweetStreaming").getOrCreate()

# Define the schema of data
schema = StructType([
    StructField("rawData", StringType(), True),
    StructField("tokens", StringType(), True),
    StructField("sentiment", DoubleType(), True),
    StructField("Category", StringType(), True)
])
#  Streaming from a directory
streaming_df = spark.readStream.option("delimiter", "\t").schema(schema).option("maxFilesPerTrigger", 1).csv("./data/output_folder_batchsize_500", header=True)

def PreprocessStream(df: DataFrame):
    """
    Preprocesses the streaming data.

    :param df: DataFrame containing the data to be preprocessed.
    :return: Preprocessed DataFrame.
    """
    # Define UDFs for cleaning and lemmatization
    def clean_text(text):
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        text = re.sub(r'\d+', '', text)
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        text = ' '.join(text.split())
        return text
    clean_text_udf = udf(clean_text, StringType())
    lemmatize_udf = udf(lambda tokens: [WordNetLemmatizer().lemmatize(word) for word in tokens], ArrayType(StringType()))


    # Apply text cleaning
    df = df.withColumn("content", F.lower(df["rawData"])).drop("rawData").drop("tokens")
    df = df.withColumn("cleaned_content", clean_text_udf(df["content"])).drop("content")

    # Tokenization
    tokenizer = Tokenizer(inputCol="cleaned_content", outputCol="tokens")
    df = tokenizer.transform(df)
    df = df.drop("content")

    # Stop Words Removal
    stop_words = list(set(word_tokenize(' '.join(stopwords.words('english')))))
    stop_words.extend(['russian', 'u'])
    remover = StopWordsRemover(inputCol="tokens", outputCol="filtered_tokens", stopWords=stop_words)
    df = remover.transform(df)

    # Lemmatization
    df = df.withColumn("words", lemmatize_udf(df["filtered_tokens"])).select("words")
    df = df.dropna()
    return df


def PredictStream(df: DataFrame):
    """
    Runs prediction on the streamed and preprocessed data.

    :param df: DataFrame containing the preprocessed data.
    :return: DataFrame with predictions.
    """

    return model.transform(df)

total_pre=[]
total_pred=[]
total=[]

def ProcessBatch(batch_df: DataFrame, batch_id: int):
    # Generate a timestamp at the start of batch processing
    batch_timestamp = time.time()

    # Perform preprocessing
    preprocessed_batch = PreprocessStream(batch_df)
    preprocesing_timestamp = time.time()

    # Perform prediction
    predictions_batch = PredictStream(preprocessed_batch)
    predictions_timestamp = time.time()

    # Add the latency
    predictions_batch = predictions_batch.withColumn("preprocessing_latency_ms", lit(round((preprocesing_timestamp- batch_timestamp) * 1000, 3)))
    predictions_batch = predictions_batch.withColumn("prediction_latency_ms", lit(round((predictions_timestamp- preprocesing_timestamp) * 1000, 3)))
    predictions_batch = predictions_batch.withColumn("total_latency_ms", lit(round((predictions_timestamp- batch_timestamp) * 1000, 3)))

    # Write to console
    predictions_batch.select("words", "features", "prediction", "preprocessing_latency_ms", "prediction_latency_ms", "total_latency_ms") \
            .write.format("console").save()
    total_pre.append(round((preprocesing_timestamp- batch_timestamp) * 1000, 3))
    total_pred.append(round((predictions_timestamp- preprocesing_timestamp) * 1000, 3))
    total.append((round((predictions_timestamp- batch_timestamp) * 1000, 3)))
    print(total_pre)
    print(total_pred)
    print(total)
    avg_pre=np.mean(total_pre)
    avg_pred=np.mean(total_pred)
    avg=np.mean(total)
    print(avg_pre)
    print(avg_pred)
    print(avg)

print("Nithya")
query = streaming_df.writeStream.foreachBatch(ProcessBatch).start()
print("fff")
query.awaitTermination(600) # Waits for 10 minutes
query.stop()


def plot_cost_history(cost_history):
    plt.figure()
    # plt.plot([20,200,1000,2000,4000,10000], cost_history)
    plt.plot([1,10,50,100,200,500], cost_history)
    plt.xlabel("batch size")
    plt.ylabel('total latency(in ms)')
    plt.show()

cost_history=[428.2888999999999,430.2045,441.5478,461.62965,465.2168500000001,513.115]
plot_cost_history(cost_history)


def plot_cost_history(cost_history):
    plt.figure()
    # plt.plot([20,200,1000,2000,4000,10000], cost_history)
    plt.plot([1,10,50,100,200,500], cost_history)
    plt.xlabel("batch size")
    plt.ylabel('prediction latency(in ms)')
    plt.show()

cost_history=[122.96055000000001,120.93915000000001,146.48485,135.60340000000002 ,149.78695,140.68965000000003]
plot_cost_history(cost_history)


def plot_cost_history(cost_history):
    plt.figure()
    # plt.plot([20,200,1000,2000,4000,10000], cost_history)
    plt.plot([1,10,50,100,200,500], cost_history)
    plt.xlabel("batch size")
    plt.ylabel('Preprocessing latency(in ms)')
    plt.show()

cost_history=[305.32830000000007,309.26539999999994,295.06305000000003 ,326.02635 ,315.43,372.42525]
plot_cost_history(cost_history)

