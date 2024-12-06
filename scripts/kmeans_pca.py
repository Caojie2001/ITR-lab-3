from pyspark.sql import SparkSession
from pyspark.ml.clustering import KMeans
from pyspark.sql.functions import monotonically_increasing_id
import pandas as pd

spark = SparkSession.builder.appName("KMeansOnPCAResults").getOrCreate()

df_original = spark.read.csv("/home/caojie2001/Data Mining/week 3/march_article.csv", header=True, inferSchema=True)
df_original = df_original.select("content", "year").withColumn("id", monotonically_increasing_id())

df_pca = spark.read.parquet("/home/caojie2001/Data Mining/week 3/pca_results.parquet")
df_pca = df_pca.withColumn("id", monotonically_increasing_id())

kmeans = KMeans(featuresCol="pcaFeatures", k=2, seed=1)
model = kmeans.fit(df_pca)
predictions = model.transform(df_pca).select("id", "prediction")

df_result = df_original.join(predictions, on="id", how="inner").select("content", "year", "prediction")

df_pandas = df_result.toPandas()

cluster_0_samples = df_pandas[df_pandas["prediction"] == 0]["content"].sample(20, random_state=1).tolist()
cluster_1_samples = df_pandas[df_pandas["prediction"] == 1]["content"].sample(20, random_state=1).tolist()

print("Cluster 0 Representative Texts:\n", cluster_0_samples)
print("\nCluster 1 Representative Texts:\n", cluster_1_samples)

df_result.write.csv("/home/caojie2001/Data Mining/week 3/kmeans_clustered_data.csv", header=True, mode="overwrite")

spark.stop()
