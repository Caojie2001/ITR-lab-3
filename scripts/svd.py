import numpy as np
from scipy.sparse import csr_matrix
from pyspark.sql import SparkSession, Row
from pyspark.mllib.linalg import Vectors as MllibVectors
from pyspark.mllib.linalg.distributed import RowMatrix
from pyspark.mllib.feature import StandardScaler

spark = SparkSession.builder.appName("SVD_Dimension_Reduction").getOrCreate()

npzfile = np.load("/home/caojie2001/Data Mining/week 3/rmrb_march_tfidf_matrix.npz")
sparse_matrix = csr_matrix((npzfile['data'], npzfile['indices'], npzfile['indptr']), shape=npzfile['shape'])

dense_matrix = sparse_matrix.toarray()

vectors_rdd = spark.sparkContext.parallelize([MllibVectors.dense(row) for row in dense_matrix])

scaler = StandardScaler(withMean=True, withStd=True)
scaler_model = scaler.fit(vectors_rdd)
scaled_rdd = scaler_model.transform(vectors_rdd)

mat = RowMatrix(scaled_rdd)

svd = mat.computeSVD(100, computeU=True)

U = svd.U
U_rdd = U.rows.map(lambda row: Row(features=MllibVectors.dense(row.toArray())))

U_df = U_rdd.toDF()

U_df.write.parquet("svd_U_100dim.parquet")

spark.stop()
