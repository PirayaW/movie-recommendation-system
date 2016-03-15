import os, os.path
import cPickle, gzip
import imdb
import csv
import re
import numpy as np
import bs4
import requests
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction import text
from sklearn.metrics.pairwise import cosine_similarity

class Recommendation_phase2:

	def __init__(self, percent):
		self.percent = percent

		self.isUpdate = False

		self.tfidf_lsa_vec = None
		self.tfidf_text_rank_vec = None
		self.tfidf_lex_rank_vec = None
		self.tfidf_luhn_vec = None
		self.setTfidfVector()

		self.movie_id = []
		self.tfidf_lsa_data = None
		self.tfidf_text_rank_data = None
		self.tfidf_lex_rank_data = None
		self.tfidf_luhn_data = None
		self.meta_data = {}
		self.setMovieData()   

		self.test_data = self.createTestSet()

	''' TFIDF '''
	def setTfidfVector(self):
		if os.path.exists('preprocess/tfidf_vec_2_' + str(self.percent) + '.pkl.gz'):
			self.setTfidfVectorFromFile()
		else:
			self.tfidf_lsa_vec = None
			self.tfidf_text_rank_vec = None
			self.tfidf_lex_rank_vec = None
			self.tfidf_luhn_vec = None

	def setTfidfVectorFromFile(self):
		print("Retriving TFIDF vector from file")
		with gzip.open('preprocess/tfidf_vec_2_' + str(self.percent) + '.pkl.gz', 'rb') as g:
			(self.tfidf_lsa_vec, self.tfidf_text_rank_vec, self.tfidf_lex_rank_vec, self.tfidf_luhn_vec) = cPickle.load(g)
		g.close() 

	def createTfidfFile(self):
		print("Creating TFIDF file")
		with gzip.open('preprocess/tfidf_vec_2_' + str(self.percent) + '.pkl.gz', 'wb') as f:
			cPickle.dump((self.tfidf_lsa_vec, self.tfidf_text_rank_vec, self.tfidf_lex_rank_vec, self.tfidf_luhn_vec), f)
		f.close()

	''' Movie Data '''
	def setMovieFromFile(self):
		print("Retriving movie data from file")
		with gzip.open('preprocess/movie_data_2_' + str(self.percent) + '.pkl.gz', 'rb') as g:
			(self.movie_id, self.tfidf_lsa_data, self.tfidf_text_rank_data, self.tfidf_lex_rank_data, self.tfidf_luhn_data, self.meta_data) = cPickle.load(g)
		g.close()

	def createDataFile(self):
		print("Creating movie data file")
		with gzip.open('preprocess/movie_data_2_' + str(self.percent) + '.pkl.gz', 'wb') as f:
			cPickle.dump((self.movie_id, self.tfidf_lsa_data, self.tfidf_text_rank_data, self.tfidf_lex_rank_data, self.tfidf_luhn_data, self.meta_data), f)
		f.close()

	def setMovieData(self):
		if os.path.exists('preprocess/movie_data_2_' + str(self.percent) + '.pkl.gz'):
			self.setMovieFromFile()
		else:
			''' get list of movie '''
			lsa_list = []
			text_rank_list = []
			lex_rank_list = []
			luhn_list = []
			ids = []
			with open('csv/plotSummary_' + str(self.percent) + '.csv') as f:
				cf = csv.reader(f)
				index = 0
				for row in cf:
					# if it is duplicate, discard
					if row[0] not in self.meta_data:
						lsa = row[2]
						lsa_cleaned = self.clean(lsa)
						lsa_list.append(lsa_cleaned)
						text_rank = row[3]
						text_rank_cleaned = self.clean(text_rank)
						text_rank_list.append(text_rank_cleaned)
						lex_rank = row[4]
						lex_rank_cleaned = self.clean(lex_rank)
						lex_rank_list.append(lex_rank_cleaned)
						luhn = row[5]
						luhn_cleaned = self.clean(luhn)
						luhn_list.append(luhn_cleaned)
						id = row[0]
						while len(id) < 7:
							id = str(0) + id
						ids.append(id)
						self.movie_id.append(id)
						self.meta_data[id] = {'title': row[1],
												'rec': set(row[6].split('|')),
												'index': index}
						index = index + 1
			f.close()

			''' create tfidf vector '''
			self.tfidf_lsa_vec = TfidfVectorizer(stop_words='english')
			self.tfidf_text_rank_vec = TfidfVectorizer(stop_words='english')
			self.tfidf_lex_rank_vec = TfidfVectorizer(stop_words='english')
			self.tfidf_luhn_vec = TfidfVectorizer(stop_words='english')

			self.tfidf_lsa_data = self.tfidf_lsa_vec.fit_transform(lsa_list)
			self.tfidf_text_rank_data = self.tfidf_text_rank_vec.fit_transform(text_rank_list)
			self.tfidf_lex_rank_data = self.tfidf_lex_rank_vec.fit_transform(lex_rank_list)
			self.tfidf_luhn_data = self.tfidf_luhn_vec.fit_transform(luhn_list)

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
			# top.append(maxIndex)
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

	def calculateScoreAll(self, matrix, n):
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
