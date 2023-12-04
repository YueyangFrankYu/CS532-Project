import nltk
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
streaming_df = spark.readStream \
    .option("delimiter", "\t") \
    .schema(schema) \
    .option("maxFilesPerTrigger", 1) \
    .csv("./data/data_test", header=True)


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

query = streaming_df.writeStream.foreachBatch(ProcessBatch).start()
query.awaitTermination(600) # Waits for 10 minutes
query.stop()