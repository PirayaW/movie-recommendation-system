import Recommendation_phase2 as rec
import numpy as np
import imdb
import time

''' run '''
x = rec.Recommendation_phase2(10)		# put 10, 20, 30, .. : percentage of top sentences extracted from the original plot
print("#Words in TFIDF (LSA): " + str(x.tfidf_lsa_data.shape[1]))
print("#Words in TFIDF (Text Rank): " + str(x.tfidf_text_rank_data.shape[1]))
print("#Words in TFIDF (Lex Rank): " + str(x.tfidf_lex_rank_data.shape[1]))
print("#Words in TFIDF (Luhn): " + str(x.tfidf_luhn_data.shape[1]))
print("#All data:" + str(x.tfidf_lsa_data.shape[0]))
print("#Test data: " + str(len(x.test_data)))

print('TFIDF all train data (1-5)')
for i in range(1, 6):
	print("Top " + str(i))
	print(x.calculateScoreAll(x.tfidf_lsa_data, i))
	print(x.calculateScoreAll(x.tfidf_text_rank_data, i))
	print(x.calculateScoreAll(x.tfidf_lex_rank_data, i))
	print(x.calculateScoreAll(x.tfidf_luhn_data, i))

printAt = 3	# print rec movies at n = ?. If you don't want to print, use n = 0

print('TFIDF test data (1-5)')
for i in range(1, 6):
	print("====================== Top " + str(i) + "============================")
	print("LSA")
	start = time.time()
	pres = []
	for id in x.test_data:
		isVerbose = i == printAt
		score = x.calculateScore(id, x.tfidf_lsa_data[x.meta_data[id]['index']], x.tfidf_lsa_data, i, verbose=False)
		if score > -1:
			pres.append(score)
	print(np.mean(np.array(pres)), float(len(pres))/len(x.test_data))
	print("Computation time: %s" %(time.time()-start))
	print("---------------------------------------------------------------------")
	print("Text Rank")
	start = time.time()
	pres = []
	for id in x.test_data:
		isVerbose = i == printAt
		score = x.calculateScore(id, x.tfidf_text_rank_data[x.meta_data[id]['index']], x.tfidf_text_rank_data, i, verbose=False)
		if score > -1:
			pres.append(score)
	print(np.mean(np.array(pres)), float(len(pres))/len(x.test_data))
	print("Computation time: %s" %(time.time()-start))
	print("---------------------------------------------------------------------")
	print("Lex Rank")
	start = time.time()
	pres = []
	for id in x.test_data:
		isVerbose = i == printAt
		score = x.calculateScore(id, x.tfidf_lex_rank_data[x.meta_data[id]['index']], x.tfidf_lex_rank_data, i, verbose=False)
		if score > -1:
			pres.append(score)
	print(np.mean(np.array(pres)), float(len(pres))/len(x.test_data))
	print("Computation time: %s" %(time.time()-start))
	print("---------------------------------------------------------------------")
	print("Luhn")
	start = time.time()
	pres = []
	for id in x.test_data:
		isVerbose = i == printAt
		score = x.calculateScore(id, x.tfidf_luhn_data[x.meta_data[id]['index']], x.tfidf_luhn_data, i, verbose=True)
		if score > -1:
			pres.append(score)
	print(np.mean(np.array(pres)), float(len(pres))/len(x.test_data))
	print("Computation time: %s" %(time.time()-start))
	print("---------------------------------------------------------------------")

x.finalize()

