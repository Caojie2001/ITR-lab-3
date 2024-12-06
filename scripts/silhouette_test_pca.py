from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.ml.evaluation import ClusteringEvaluator

spark = SparkSession.builder.appName("KMeans").getOrCreate()

df_pca = spark.read.parquet("pca_results.parquet")

max_k = 20
silhouette_scores = []

for k in range(2, max_k + 1):
    kmeans = KMeans(featuresCol="pcaFeatures", k=k, seed=1)
    model = kmeans.fit(df_pca)
    
    predictions = model.transform(df_pca)
    evaluator = ClusteringEvaluator(featuresCol="pcaFeatures", metricName="silhouette")
    silhouette_score = evaluator.evaluate(predictions)
    silhouette_scores.append((k, silhouette_score))
    print(f"For k={k}, the Silhouette Score is {silhouette_score}")

print("\nSilhouette Scores for K values from 2 to 20:")
for k, score in silhouette_scores:
    print(f"K={k}: Silhouette Score = {score}")

spark.stop()
