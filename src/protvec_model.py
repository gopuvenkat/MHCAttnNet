import sys, os, json
import argparse

import numpy as np
from gensim.models import word2vec
from Bio import SeqIO
from gensim.models.keyedvectors import KeyedVectors
from tqdm import tqdm
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

def split_ngrams(seq, n):
	"""
	Split sequence into non-overlapping n-grams
	'ATSKLGH' --> [['ATS','KLG'],['TSK','LGH],['SKL']]
	"""
	kmers = list()
	for i in range(n):
		kmers.append(zip(*[iter(seq[i:])]*n))
	str_ngrams = list()
	for ngrams in kmers:
		x = list()
		for ngram in ngrams:
			x.append("".join(ngram))
		str_ngrams.append(x)
	return str_ngrams

def generate_corpusfile(corpus_fname, n, out):
	with open(out, "w") as f:
		for r in tqdm(SeqIO.parse(corpus_fname, "fasta")):
			if("B" not in r.seq and "J" not in r.seq and "O" not in r.seq and "U" not in r.seq and "Z" not in r.seq):	# sanity check to remove invalid amino acids
				ngram_patterns = split_ngrams(r.seq, n)
				for ngram_pattern in ngram_patterns:
					f.write(" ".join(ngram_pattern) + "\n")	# Take all the sequences and split them into kmers

class ProtVec(word2vec.Word2Vec):
	def __init__(self, corpus_fname=None, n=3, size=100, out="output_corpus.txt", sg=1, window=25, min_count=1, workers=9):
		self.n = n
		self.size = size
		self.corpus_fname = corpus_fname
		self.sg = sg
		self.window = window
		self.workers = workers
		self.out = out
		self.vocab = min_count

		if(corpus_fname is not None):
			if(not os.path.isfile(out)):
				print("-- Generating corpus --")
				generate_corpusfile(corpus_fname, n, out)
			else:
				print("-- Corpus File Found --")
		
		self.corpus = word2vec.Text8Corpus(out)
		print("-- Corpus Setup Successful --")

	def word2vec_init(self, vectors_txt, model_weights):
		print("-- Initializing Word2Vec model --")
		print("-- Training the model --")
		self.m = word2vec.Word2Vec(self.corpus, size=self.size, sg=self.sg, window=self.window, min_count=self.vocab, workers=self.workers)
		self.m.wv.save_word2vec_format(vectors_txt)
		self.m.save(model_weights)
		print("-- Saving Model Weights to : %s " % (vectors_txt))

	def load_protvec(self, model_weights):
		print("-- Load Word2Vec model --")
		self.m = word2vec.Word2Vec.load(model_weights)
		return self.m

def tsne_plot(model, n_components=2, random_state=42):
	"""
	Create a TSNE model and plot it
	"""
	print("-- Start t-SNE plot --")
	labels = []
	tokens = []
	
	for word in model.wv.vocab:
		tokens.append(model[word])
		labels.append(word)

	tsne_model = TSNE(n_components=n_components, random_state=random_state)
	new_values = tsne_model.fit_transform(tokens)

	x = []
	y = []
	for value in new_values:
		x.append(value[0])
		y.append(value[1])
	
	plt.figure(figsize=(16, 16))
	for i in range(len(x)):
		plt.scatter(x[i], y[i])
		plt.annotate(labels[i], xy=(x[i], y[i]), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom")
	plt.show()

if __name__ == "__main__":
	parser = argparse.ArgumentParser(description="Train ProtVec using gensim (Word2Vec)")
	parser.add_argument("-f", "--corpus_fname", default="../data/uniprot_sprot.fasta", help="path to the input (FASTA) corpus")
	parser.add_argument("-o", "--output_corpus", default="../data/output_corpus_3.txt", help="path to the output corpus")
	parser.add_argument("-n", "--ngram_length", type=int, default=3, help="ngram length")
	parser.add_argument("--skip_gram", type=int, default=1, help="to enable skip-gram algorithm")
	parser.add_argument("--window", type=int, default=25, help="set window size")
	parser.add_argument("--min_count", type=int, default=1, help="neglect those words whose frequency is less than this threshold")
	parser.add_argument("--workers", type=int, default=12)
	parser.add_argument("-s", "--size", type=int, default=100, help="embedding dimension")
	parser.add_argument("-v", "--vectors", default="../data/3-gram-vectors.txt", help="path to the text file where the vectors are to be stored")
	parser.add_argument("-m", "--model_weights", default="../data/3-gram-model-weights.mdl", help="path to the binary file where the model weights are to be stored")
	args = parser.parse_args()

	# model = ProtVec(corpus_fname=args.corpus_fname, n=args.ngram_length, out=args.output_corpus, sg=args.skip_gram, window=args.window, min_count=args.min_count, workers=args.workers)
	# model.word2vec_init(vectors_txt=args.vectors, model_weights=args.model_weights)

	model = ProtVec()
	model.load_protvec(model_weights=args.model_weights)

	tsne_plot(model.m, 3)

	w1 = "WFN"
	print("Most similar to WFN", model.m.wv.most_similar(positive=w1))
	
	w1 = "SQQ"
	print("Most similar to SQQ", model.m.wv.most_similar(positive=w1))