import RecommendationController as rec
import numpy as np
import imdb
import time

''' run '''
x = rec.RecommendationController()		
print("#Words in TFIDF: " + str(x.tfidf_data.shape[1]))
print("#All data: " + str(x.tfidf_data.shape[0]))
print("#Test data: " + str(len(x.test_data)))

print('TFIDF all train data (1-5)')
for i in range(1, 6):
	print("Top " + str(i))
	print(x.calcuateScoreAll(x.tfidf_data, i))

print('SVD all train data (1-5)')
for i in range(1, 6):
	print("Top " + str(i))
	print(x.calcuateScoreAll(x.svd_data, i))

printAt = 3		# print rec movies at n = ?. If you don't want to print, use n = 0

print('TFIDF test data (1-5)')
for i in range(1, 6):
	print("====================== Top " + str(i) + "============================")
	start = time.time()
	pres = []
	for id in x.test_data:
		isVerbose = i == printAt
		score = x.calculateScore(id, x.tfidf_data[x.meta_data[id]['index']], x.tfidf_data, i, verbose=isVerbose)
		if score > -1:
			pres.append(score)
	print(np.mean(np.array(pres)), float(len(pres))/len(x.test_data))
	print("Computation time: %s" %(time.time()-start))
	print("---------------------------------------------------------------------")

print('SVD test data (1-5)')
for i in range(1, 6):
	print("====================== Top " + str(i) + "============================")
	start = time.time()
	pres = []
	for id in x.test_data:
		isVerbose = i == printAt
		score = x.calculateScore(id, x.svd_data[x.meta_data[id]['index']].reshape(1, -1), x.svd_data, i, verbose=isVerbose)
		if score > -1:
			pres.append(score)
	print(np.mean(np.array(pres)), float(len(pres))/len(x.test_data))
	print("Computation time: %s" %(time.time()-start))
	print("---------------------------------------------------------------------")

x.finalize()

