import RecommendationController as rec
import RecommendationController_NE as ne
from sklearn.decomposition import TruncatedSVD
import matplotlib.pyplot as plt
import os, os.path

x = rec.RecommendationController()
svd_vec = TruncatedSVD()	# default n_components = 2
print(x.tfidf_data)
svd_data = svd_vec.fit_transform(x.tfidf_data)
plt.plot(svd_data[:,0], svd_data[:,1], 'ro')
path = 'pic/'
if not os.path.exists(path):
		os.makedirs(path)
filename = 'recCluster.png'
plt.savefig(path + filename)
plt.show()

y = ne.RecommendationController_NE()
print(y.tfidf_vec.get_feature_names())
svd_vec = TruncatedSVD()
svd_data = svd_vec.fit_transform(y.tfidf_data)
plt.plot(svd_data[:,0], svd_data[:,1], 'ro')
filename = 'neCluster.png'
plt.savefig(path + filename)
plt.show()