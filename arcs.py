import nltk
import pandas as pd
from scipy import fftpack
import matplotlib.pyplot as plt 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import summarise as summ


plt.style.use('ggplot')

def de_gutenberger(filename):
	# Gets rid of the Gutenberg header and footer crap
	with open(filename, "r", encoding='utf-8') as f:
		data = f.readlines()

	for index1, value1 in enumerate(data):
		if u"START OF THIS PROJECT GUTENBERG" in value1:
			break
	for index2, value2 in enumerate(data):
	
		if u"END OF THIS PROJECT GUTENBERG" in value2:
			break

	if index1 == index2:
		index1 = 0
		index2 = len(data)

	output_string = ""
	for i in data[index1 + 25:index2-1]:
		output_string += i

	return output_string


def words(input_str):

	split_words = nltk.word_tokenize(input_str.lower())
	d = {'word' : split_words}

	return pd.DataFrame(data=d)


def sentences(input_str):

	split_sent = nltk.sent_tokenize(input_str)

	d = {'sentences' : split_sent}

	return pd.DataFrame(data=d)


def word_sentiment(df):

	df_w = pd.read_csv('texts/labMT.txt', delimiter='\t')
	df_w['Sentiment'] = ((df_w.happiness_average - 
		df_w.happiness_average.mean()) / df_w.happiness_average.std())

	df['Freq'] = df.groupby('word')['word'].transform('count')

	df = df.merge(df_w[['word', 'Sentiment']], 
		on='word', how='left').fillna(0)
	
	df['Sentiment'] = df.H_norm / df.Freq
	
	return df


def sentence_sentiment(df):

	sid = SentimentIntensityAnalyzer()
	df['Sentiment'] = df.sentences.apply(lambda x: 
			sid.polarity_scores(x)['compound'])

	return df


def plot_arc(df, components):

	sample_freq = fftpack.fftfreq(len(df), d=1/len(df))
	sig_fft = fftpack.fft(df.Sentiment.values)

	pidxs = np.where(sample_freq > 0)
	freqs = sample_freq[pidxs]
	power = np.abs(sig_fft)[pidxs]

	freq = freqs[np.argsort(power)[::-1][components]]

	sig_fft[np.abs(sample_freq) > freq] = 0
	main_sig = fftpack.ifft(sig_fft)

	plt.plot(100*np.arange(len(df)) / len(df), main_sig)
	plt.xlabel('% of book')
	plt.title('Portrait of the Artist')
	
	plt.show()


directory = 'texts/'
book = 'Great_Expectations_by_Charles_Dickens_.txt'
book = 'portrait_of_the_artist.txt' 

raw_text = de_gutenberger(directory + book)

df = sentences(raw_text)
df = sentence_sentiment(df)

plot_arc(df, 0)

region = float(input('Select a region to be printed \n'))
reg_ind = int(region / 100 * len(df))

FS = summ.FrequencySummarizer()
text_region = ' '.join(df.sentences[reg_ind-10:reg_ind+10].values)

summary = FS.summarize(text_region, 3)

print('Text Summary \n:' + '-'*20)
[print(s+'\n'+20*'-'+'\n') for s in summary]


