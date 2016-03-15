import RecommendationController as rec

def getType():
	type = raw_input("Movie name (m) / Description (d) / Exit (e): ")
	if type == 'm' or type == 'M':
		recommendByMovie()
	elif type == 'd' or type == 'D':
		recommendByDescription()
	elif type == 'e' or type == 'E':
		pass
	else:
		print("Invalid input. Please try again.")
		getType()

def recommendByMovie():		
	user_input = raw_input("Please enter movie name: ")
	movie = x.ia.search_movie(user_input)
	if len(movie) > 0:
		movie = movie[0]
		movie = x.ia.get_movie(movie.movieID)
		id = movie.movieID
		title = movie.get('title')
		if id in x.movie_id:
			recs = x.findTopSimilarMovies(x.tfidf_data[x.meta_data[id]['index']], x.tfidf_data, 5)
			printRecommendMovies(recs)
		else:
			plot = movie.get('plot')
			plot = " ".join(plot)
			if plot is not None:
				getRecommendMovies(plot)
			else:
				print("No related movies for your movies. Please try again.")
				getType()
	else:
		print("No related movies for your movies. Please try again.")
		getType()

def recommendByDescription():	
	user_input = raw_input("Please enter movie description: ")
	getRecommendMovies(user_input)

def getRecommendMovies(plot):
	tfidf = x.tfidf_vec.transform([x.clean(plot)])
	if tfidf.sum() == 0:
		print("Sorry, not enough information.")
		getType()
	else:
		recs = x.findTopSimilarMovies(tfidf, x.tfidf_data, 5)
		printRecommendMovies(recs)

def printRecommendMovies(recs):
	for i in recs:
		m = x.ia.get_movie(i)
		p = m.get('plot')
		p = " ".join(p)
		print(m.get('title') + " (ID: " + str(i) + ")")
	if len(recs) == 0:
		print("No related movies for your movies. Please try again.")
		getType()
	else:
		again = raw_input("Search another movies? (y/n): ")
		if again == 'Y' or again == 'y':
			getType()
		else:
			print("Bye!")

x = rec.RecommendationController()	
getType()

