# ITR-lab-3

# Repository Structure

This repository is organized as follows:

## `kmeans_clustered_data/`
Contains datasets with kmeans classification based on PCA.

## `scripts/`
Contains different scripts used in the project.

- `preprocess.ipynb`: Preprocess pipeline.
- `pca.py`: PCA dimensionality reduction on data.
- `svd.py`: SVD dimensionality reduction on data.
- `silhouette_test_raw.py`: Silhouette test of Kmeans on raw data.
- `silhouette_test_pca.py`: Silhouette test of Kmeans on PCA-processed data.
- `silhouette_test_svd.py`: Silhouette test of Kmeans on SVD-processed data.
- `kmeans_pca.py`: Kmeans on PCA-processed data.
- `temporal_analysis.ipynb`: Temporal analysis of occurences based on Kmeans.
- `pca_cgss.ipynb`: PCA analysis of the CGSS data.

## Other Files
- `ITR lab week 3.pdf`: Submission for ITR lab.
- `cgss2017.csv`: CGSS 2017 dataset, which is used for PCA analysis.
- `kmeans_output.txt`: Representative samples of the kmeans classification result. Rather messy, please refer to `kmeans_clustered_data/` for detailed results.
- `silhouette_results.txt`: Output of the silhouette tests results for raw data, PCA-processed data and SVD-processed data.
- `README.md`: introduction of the repository.

