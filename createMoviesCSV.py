import csv
import imdb
import imdb.parser.http as http
import bs4
import requests

ia = imdb.IMDb() 
ih = http.IMDbHTTPAccessSystem(ia)
movie_ID =[]
movie_plot = []
movie_title = []
movie_rec = []

#get movie from keyword
movie = ia.search_movie("hunger games")
movie = movie[0] #first result
id= movie.movieID
while len(id) < 7:
	id= str(0) + id
movie = ia.get_movie(movie.movieID)
title = movie.get('title').encode('ascii', 'replace')
print title
src = requests.get('http://www.imdb.com/title/tt' + id + '/').text
bs = bs4.BeautifulSoup(src, "lxml")
recs = [rec['data-tconst'][2:] for rec in bs.findAll('div', 'rec_item')]
plotlist = movie.get('plot')
if plotlist is not None:
	movie_ID.append(str(id))
	movie_title.append(title)
	plot = " ".join(plotlist)
	plot = plot.encode('ascii', 'replace') 
	movie_plot.append(plot)
	movie_rec.append('|'.join(recs))
else:
	plot = '' 

csv_out = open('csv/mymovies.csv', 'a')
mywriter = csv.writer(csv_out)
rows = zip(movie_ID, movie_title, movie_plot, movie_rec)
mywriter.writerows(rows)
csv_out.close()
