import numpy as np
import pandas as pd
import umap.umap_ as umap
from sklearn.mixture import GaussianMixture

features = pd.read_csv('bert-combined-features.csv', delimiter=',', header=None) # load the features after creating them

#UMAP
reducer = umap.UMAP(
        n_neighbors=45,
        min_dist=0.7,
        n_components=2,
        metric='manhattan'
    )

low_dim_features = reducer.fit_transform(features)

# GMM
gmm = GaussianMixture(n_components=2, covariance_type='full').fit(low_dim_features)
predictions = gmm.predict(low_dim_features)
probs = gmm.predict_proba(low_dim_features)

np.savetxt("gmm_combined_prob.csv", probs, delimiter=',')