import os, os.path
import cPickle, gzip
import imdb
import csv
import re
import numpy as np
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity
import math

class Recommendation_phase3:

	def __init__(self):
		self.chunkRatio = [0.3, 0.4, 0.3]
		self.isUpdate = False

		self.tfidf_vec = [None for i in range(0,len(self.chunkRatio))]
		self.setTfidfVector()

		self.movie_id = []
		self.tfidf_data = [None for i in range(0,len(self.chunkRatio))]
		self.meta_data = {}
		self.setMovieData()   

		self.test_data = self.createTestSet()

	''' TFIDF '''
	def setTfidfVector(self):
		if os.path.exists('preprecess/tfidf_vec_phase3.pkl.gz'):
			self.setTfidfVectorFromFile()
		else:
			self.tfidf_vec = [None for i in range(0,len(self.chunkRatio))]

	def setTfidfVectorFromFile(self):
		print("Retriving TFIDF vector from file")
		with gzip.open('preprocess/tfidf_vec_3.pkl.gz', 'rb') as g:
			self.tfidf_vec = cPickle.load(g)
		g.close() 

	def createTfidfFile(self):
		print("Creating TFIDF file")
		with gzip.open('preprocess/tfidf_vec_3.pkl.gz', 'wb') as f:
			cPickle.dump(self.tfidf_vec, f)
		f.close()

	''' Movie Data '''
	def setMovieFromFile(self):
		print("Retriving movie data from file")
		with gzip.open('preprocess/movie_data_3.pkl.gz', 'rb') as g:
			(self.movie_id, self.tfidf_data, self.meta_data) = cPickle.load(g)
		g.close()

	def createDataFile(self):
		print("Creating movie data file")
		with gzip.open('preprocess/movie_data_3.pkl.gz', 'wb') as f:
			cPickle.dump((self.movie_id, self.tfidf_data, self.meta_data), f)
		f.close()

	def setMovieData(self):
		if os.path.exists('preprocess/movie_data_3.pkl.gz'):
			self.setMovieFromFile()
		else:
			''' get list of movie '''
			ids = []
			chunk = [[], [], []]
			with open('csv/mymovies.csv') as f:
				cf = csv.reader(f)
				index = 0
				for row in cf:
					# if it is duplicate, discard
					if row[0] not in self.meta_data:
						plot = row[2]
						sentences = plot.split('.')
						n = len(sentences)
						# print("=========" + row[1] + "========")
						start = 0
						for i in range(len(self.chunkRatio)):
							m = int(n*self.chunkRatio[i])
							end = start + m
							if i == len(self.chunkRatio):
								end = n
							temp = sentences[start:end]
							# print(str(temp) + "\n")
							start = end
							cleaned = self.clean(" ".join(temp))
							chunk[i].append(cleaned)
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
			for i in range(len(self.chunkRatio)):
				self.tfidf_vec[i] = TfidfVectorizer(stop_words='english')
				self.tfidf_vec[i].fit(chunk[i])
				self.tfidf_data[i] = self.tfidf_vec[i].transform(chunk[i])

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

	def findAllTopSimilarMovies(self, n):
		m = len(self.chunkRatio)
		w = self.tfidf_data[0].shape[0]
		sim = [0 for i in range(0, m)]
		avg_sim = np.zeros((w, w))
		for i in range(m):
			sim[i] = cosine_similarity(self.tfidf_data[i], self.tfidf_data[i])
			avg_sim = avg_sim + np.array(sim[i])
		avg_sim = avg_sim/m
		top = []
		for i in range(len(avg_sim)):
			avg_sim[i][i] = -1
			if np.array([self.tfidf_data[j][i].sum() for j in range(0,m)]).sum() == 0:
				top.append([])
			else:
				top.append(self.topIndex(avg_sim[i], n))
		return top

	# return "id" of the movie
	def findTopSimilarMovies(self, vec, matrix, n):
		if vec.sum(axis=1) == 0:
			return []
		sim = cosine_similarity(vec, matrix)
		return self.topIndex(sim[0], n)

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

	def calculateScore(self, n, verbose=False):
		topId = self.findAllTopSimilarMovies(n)
		precision = []
		for i in range(len(self.movie_id)):
			count = 0.0
			id = self.movie_id[i]
			if id in self.test_data and len(self.meta_data[id]['rec']) >= n:
				for tid in topId[i]:
					if tid in self.meta_data[id]['rec']:
						count = count + 1
				if len(topId[i]) > 0:
					precision.append(count/len(topId[i]))
					if verbose:
						print(self.meta_data[id]['title'] + ': ' + str([self.meta_data[j]['title'] for j in topId[i]]) + str(count/len(topId[i])))
		return np.mean(np.array(precision))

	def calculateScoreAll(self, n):
		topId = self.findAllTopSimilarMovies(n)
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
		return np.mean(np.array(precision))

	def finalize(self):
		if self.isUpdate:
			self.createTfidfFile()
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
