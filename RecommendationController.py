import os, os.path
import cPickle, gzip
import csv
import re
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

class RecommendationController:

	def __init__(self):
		self.isUpdate = False

		self.tfidf_vec = None
		self.setTfidfVector()

		self.svd_vec = None
		self.setSVDVector()

		self.movie_id = []
		self.svd_data = None
		self.tfidf_data = None
		self.meta_data = {}
		self.setMovieData()   

		self.test_data = self.createTestSet()

	''' TFIDF '''
	def setTfidfVector(self):
		if os.path.exists('preprocess/tfidf_vec.pkl.gz'):
			self.setTfidfVectorFromFile()
		else:
			self.tfidf_vec = None

	def setTfidfVectorFromFile(self):
		print("Retriving TFIDF vector from file")
		with gzip.open('preprocess/tfidf_vec.pkl.gz', 'rb') as g:
			self.tfidf_vec = cPickle.load(g)
		g.close() 

	def createTfidfFile(self):
		print("Creating TFIDF file")
		with gzip.open('preprocess/tfidf_vec.pkl.gz', 'wb') as f:
			cPickle.dump(self.tfidf_vec, f)
		f.close()

	''' SVD '''
	def setSVDVector(self):
		if os.path.exists('preprocess/svd_vec.pkl.gz'):
			self.setSvdVectorFromFile()
		else:
			self.svd_vec = None

	def setSvdVectorFromFile(self):
		print("Retriving SVD vector from file")
		with gzip.open('preprocess/svd_vec.pkl.gz', 'rb') as g:
			self.svd_vec = cPickle.load(g)
		g.close()

	def createSvdFile(self):
		print("Creating SVD file")
		with gzip.open('preprocess/svd_vec.pkl.gz', 'wb') as f:
			cPickle.dump(self.svd_vec, f)
		f.close()

	''' Movie Data '''
	def setMovieFromFile(self):
		print("Retriving movie data from file")
		with gzip.open('preprocess/movie_data.pkl.gz', 'rb') as g:
			(self.movie_id, self.tfidf_data, self.svd_data, self.meta_data) = cPickle.load(g)
		g.close()

	def createDataFile(self):
		print("Creating movie data file")
		with gzip.open('preprocess/movie_data.pkl.gz', 'wb') as f:
			cPickle.dump((self.movie_id, self.tfidf_data, self.svd_data, self.meta_data), f)
		f.close()

	def setMovieData(self):
		if os.path.exists('preprocess/movie_data.pkl.gz'):
			self.setMovieFromFile()
		else:
			''' get list of movie '''
			plots = []
			ids = []
			with open('csv/mymovies.csv') as f:
				cf = csv.reader(f)
				index = 0
				for row in cf:
					# if it is duplicate, discard
					if row[0] not in self.meta_data:
						plot = row[2]
						cleaned = self.clean(plot)
						plots.append(cleaned)
						id = row[0]
						while len(id) < 7:
							id = str(0) + id
						ids.append(id)
						self.movie_id.append(id)
						self.meta_data[id] = {'title': row[1],
												'rec': set(row[3].split('|')),
												'index': index}
						index = index + 1
			f.close()

			''' create tfidf vector '''
			self.tfidf_vec = TfidfVectorizer(stop_words='english')
			self.tfidf_vec.fit(plots)
			self.tfidf_data = self.tfidf_vec.transform(plots)

			''' create svd vector '''
			self.svd_vec = TruncatedSVD(n_components=100, random_state=42)
			self.svd_vec.fit(self.tfidf_data)
			self.svd_data = self.svd_vec.transform(self.tfidf_data)
			self.isUpdate = True

	def clean(self, data):
		stemmer = SnowballStemmer("english", ignore_stopwords=True)
		temp = data
		temp = re.sub("[,.-:/()?{}*$#&]"," ",temp)
		temp = temp.lower()
		words = temp.split()
		after_stem = [stemmer.stem(plural) for plural in words]
		temp = " ".join(after_stem)
		return temp

	# return "id" of the movie
	def findTopSimilarMovies(self, vec, matrix, n):
		if vec.sum(axis=1) == 0:
			return []
		sim = cosine_similarity(vec, matrix)
		return self.topIndex(sim[0], n)

	# return "id" of the movie
	def findAllTopSimilarMovies(self, matrix, n):
		sim = cosine_similarity(matrix, matrix)
		top = []
		for i in range(len(sim)):
			sim[i][i] = -1
			if matrix[i].sum() == 0:
				top.append([])
			else:
				top.append(self.topIndex(sim[i], n))
		return top

	def topIndex(self, v, n):
		top = []
		for i in range(n):
			maxIndex = np.argmax(v)
			v[maxIndex] = -1
			top.append(self.movie_id[maxIndex])
		return top

	def getMovieTitleFromId(self, ids):
		names = []
		for id in ids:
			names.append(self.meta_data[id]['title'])
		return names

	def calculateScore(self, id, vec, matrix, n, verbose=False):
		count = 0.0
		if id not in self.meta_data:
			pass # handle it in phase 2
		else:
			if len(self.meta_data[id]['rec']) >= n:
				# discard itself
				topId = self.findTopSimilarMovies(vec, matrix, n+1)
				k = len(topId)
				for tid in topId:
					if tid == id:
						k = k-1
					if tid in self.meta_data[id]['rec']:
						count = count + 1
				if k <= 0:
					return -1
				else:
					if verbose:
						print(self.meta_data[id]['title'] + ': ' + str([self.meta_data[i]['title'] for i in topId]) + str(count/k))
					return (count/k)
			else:
				return -1

	def calcuateScoreAll(self, matrix, n):
		topId = self.findAllTopSimilarMovies(matrix, n)
		precision = []
		for i in range(len(self.movie_id)):
			count = 0.0
			id = self.movie_id[i]
			if len(self.meta_data[id]['rec']) >= n:
				for tid in topId[i]:
					if tid in self.meta_data[id]['rec']:
						count = count + 1
				if len(topId[i]) > 0:
					precision.append(count/len(topId[i]))
		return (np.mean(np.array(precision)), float(len(precision))/len(self.movie_id))

	def finalize(self):
		if self.isUpdate:
			self.createTfidfFile()
			self.createSvdFile()
			self.createDataFile()

	def getListOfRecNotInDB(self):
		s = set()
		for key, value in self.meta_data.iteritems():
			for rec in value['rec']:
				if rec != '' and rec not in self.meta_data:
					s.add(rec)
		return s

	def createTestSet(self):
		self.test_data = []
		for key, value in self.meta_data.iteritems():
			for rec in value['rec']:
				if rec not in self.meta_data:
					break
			else:
				self.test_data.append(key)
		return self.test_data
