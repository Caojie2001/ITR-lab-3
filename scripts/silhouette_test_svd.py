from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator
from pyspark.ml.functions import vector_to_array
from pyspark.ml.linalg import DenseVector

spark = SparkSession.builder.appName("KMeans").getOrCreate()

df_svd = spark.read.parquet("/home/caojie2001/Data Mining/week 3/svd_U_100dim.parquet")

df_svd = df_svd.withColumn("features_dense", vector_to_array("features"))

max_k = 20
silhouette_scores = []

for k in range(2, max_k + 1):
    kmeans = KMeans(featuresCol="features_dense", k=k, seed=1)
    model = kmeans.fit(df_svd)
    
    predictions = model.transform(df_svd)
    evaluator = ClusteringEvaluator(featuresCol="features_dense", metricName="silhouette")
    silhouette_score = evaluator.evaluate(predictions)
    silhouette_scores.append((k, silhouette_score))
    print(f"For k={k}, the Silhouette Score is {silhouette_score}")

print("\nSilhouette Scores for K values from 2 to 20:")
for k, score in silhouette_scores:
    print(f"K={k}: Silhouette Score = {score}")

spark.stop()
