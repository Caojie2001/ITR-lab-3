import numpy as np
from scipy.sparse import csr_matrix
from pyspark.sql import SparkSession
from pyspark.ml.linalg import Vectors
from pyspark.sql import Row
from pyspark.ml.feature import PCA

spark = SparkSession.builder.appName("TFIDF_PCA_Save").getOrCreate()

npzfile = np.load("/home/caojie2001/Data Mining/week 3/rmrb_march_tfidf_matrix.npz")
sparse_matrix = csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])

dense_matrix = sparse_matrix.toarray()

rows = [Row(features=Vectors.dense(row)) for row in dense_matrix]
df_spark = spark.createDataFrame(rows)

pca = PCA(k=100, inputCol="features", outputCol="pcaFeatures")
pca_model = pca.fit(df_spark)
df_pca = pca_model.transform(df_spark)

df_pca.select("pcaFeatures").show(5, truncate=False)

df_pca.select("pcaFeatures").write.parquet("pca_results.parquet")

spark.stop()
