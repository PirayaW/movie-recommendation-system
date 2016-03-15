import Recommendation_phase3 as rec

''' run '''
x = rec.Recommendation_phase3()	
for i in range(0, len(x.chunkRatio)):
	print("#Words in TFIDF (Part " + str(i+1) + "): " + str(x.tfidf_data[i].shape[1]))
print("#All data: " + str(x.tfidf_data[0].shape[0]))
print("#Test data: " + str(len(x.test_data)))

print('TFIDF all train data (1-5)')
for i in range(1, 6):
	print("Top " + str(i))
	print(x.calculateScoreAll(i))

printAt = 3		# print rec movies at n = ?. If you don't want to print, use n = 0

print('TFIDF test data (1-5)')
for i in range(1, 6):
	print("====================== Top " + str(i) + "============================")
	isVerbose = i == printAt
	print(x.calculateScore(i, verbose=isVerbose))
	print("---------------------------------------------------------------------")

x.finalize()

