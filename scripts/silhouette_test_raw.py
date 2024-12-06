import numpy as np
from scipy.sparse import csr_matrix
from pyspark.sql import SparkSession, Row
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.linalg import Vectors
from pyspark.ml.functions import vector_to_array

spark = SparkSession.builder.appName("KMeans").getOrCreate()

npzfile = np.load("/home/caojie2001/Data Mining/week 3/rmrb_march_tfidf_matrix.npz")
sparse_matrix = csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])

dense_matrix = sparse_matrix.toarray()

rows = [Row(features=Vectors.dense(row)) for row in dense_matrix]
df_tfidf = spark.createDataFrame(rows)

max_k = 20
silhouette_scores = []

for k in range(2, max_k + 1):
    kmeans = KMeans(featuresCol="features", k=k, seed=1)
    model = kmeans.fit(df_tfidf)
    
    predictions = model.transform(df_tfidf)
    evaluator = ClusteringEvaluator(featuresCol="features", metricName="silhouette")
    silhouette_score = evaluator.evaluate(predictions)
    silhouette_scores.append((k, silhouette_score))
    print(f"For k={k}, the Silhouette Score is {silhouette_score}")

print("\nSilhouette Scores for K values from 2 to 20:")
for k, score in silhouette_scores:
    print(f"K={k}: Silhouette Score = {score}")

spark.stop()
