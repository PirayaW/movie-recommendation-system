# create plot summary file

from __future__ import absolute_import
from __future__ import division, print_function, unicode_literals

import csv
import math
from sumy.parsers.plaintext import PlaintextParser
from sumy.nlp.tokenizers import Tokenizer
from sumy.summarizers.lsa import LsaSummarizer
from sumy.summarizers.text_rank import TextRankSummarizer
from sumy.summarizers.lex_rank import LexRankSummarizer
from sumy.summarizers.luhn import LuhnSummarizer
from sumy.nlp.stemmers import Stemmer
from sumy.utils import get_stop_words

LANGUAGE = "english"
percent = 50
meta_data = set()
ids = []
title = []
lsa_list = []
text_rank_list = []
lex_rank_list = []
luhn_list = []
rec = []
stemmer = Stemmer(LANGUAGE)
lsaSummarizer = LsaSummarizer(stemmer)
textRankSummarizer = TextRankSummarizer(stemmer)
lexRankSummarizer = LexRankSummarizer(stemmer)
luhnSummarizer = LuhnSummarizer(stemmer)

def countSentence(text):
	return len(text.split('.'))

with open('mymovies.csv') as f:
	cf = csv.reader(f)
	for row in cf:
		id = row[0]
		while len(id) < 7:
			id = str(0) + id
		if id not in meta_data:
			plot = row[2]
			nSentences = countSentence(plot)
			parser = PlaintextParser.from_string(plot, Tokenizer(LANGUAGE))
			lsa = [str(sentence) for sentence in lsaSummarizer(parser.document, math.floor(nSentences*(percent/100)))]
			text_rank = [str(sentence) for sentence in textRankSummarizer(parser.document, math.floor(nSentences*(percent/100)))]
			lex_rank = [str(sentence) for sentence in lexRankSummarizer(parser.document, math.floor(nSentences*(percent/100)))]
			luhn = [str(sentence) for sentence in luhnSummarizer(parser.document, math.floor(nSentences*(percent/100)))]
			meta_data.add(id)
			ids.append(id)
			title.append(row[1])
			lsa_list.append(" ".join(lsa))
			text_rank_list.append(" ".join(text_rank))
			lex_rank_list.append(" ".join(lex_rank))
			luhn_list.append(" ".join(luhn))
			rec.append(row[3])
f.close()

print("Creating plot summary (csv) file")
csv_out = open('csv/plotSummary_' + str(percent) + '.csv', 'wb')
mywriter = csv.writer(csv_out)
rows = zip(ids, title, lsa_list, text_rank_list, lex_rank_list, luhn_list, rec)
mywriter.writerows(rows)
csv_out.close()

