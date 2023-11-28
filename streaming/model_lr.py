import pyspark
import numpy as np
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml import Pipeline, PipelineModel
from pyspark.ml.feature import HashingTF, IDF, Tokenizer, StringIndexer, CountVectorizer
from pyspark.ml.classification import LogisticRegression
from pyspark.ml.evaluation import MulticlassClassificationEvaluator

spark= SparkSession.builder.getOrCreate()

path = './data_raw'
paths = [path+'/'+str(i)+'.csv' for i in range(23)]

# data load
df = spark.read.option("delimiter", "\t").csv(paths,header=True)
df = df.dropna()

(trainData, testData) = df.randomSplit([0.7, 0.3], seed = 100)
testData.write.csv('./data_test', header=True, mode="overwrite",sep="\t")

# tokenization
tokenizer = Tokenizer(inputCol="tokens", outputCol="words")
trainData = tokenizer.transform(trainData)
trainData = trainData.drop("rawData").drop("tokens")

testData2 = tokenizer.transform(testData)
testData2 = testData2.drop("rawData").drop("tokens")

# pipeline 
countVectors = CountVectorizer(inputCol="words", outputCol="cv", vocabSize=30000, minDF=5)
idf = IDF(inputCol='cv',outputCol='features',minDocFreq=5)
label = StringIndexer(inputCol = "Category", outputCol = "label")
lr = LogisticRegression()
evaluator = MulticlassClassificationEvaluator(predictionCol="prediction")
pipeline = Pipeline(stages=[label, countVectors, idf, lr])

lrModel = pipeline.fit(trainData)
predictions = lrModel.transform(testData2)

accuracy = evaluator.evaluate(predictions, {evaluator.metricName: "accuracy"})
precision = evaluator.evaluate(predictions, {evaluator.metricName: "weightedPrecision"})
recall = evaluator.evaluate(predictions, {evaluator.metricName: "weightedRecall"})

# Print the results
print("Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)

lrModel.write().overwrite().save("./model")