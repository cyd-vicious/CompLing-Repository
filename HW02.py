#Задание 1
dvach = open("C://Users//tonch//CydnieCantFigureOutDisks//2ch_corpus//2ch_corpus.txt", encoding = "utf-8").read()[:1000]
news = open("C://Users//tonch//CydnieCantFigureOutDisks//lenta//lenta.txt", encoding = "utf-8").read()[:1000]

from string import punctuation
from razdel import sentenize
from razdel import tokenize as razdel_tokenize
import numpy as np
from collections import Counter
from nltk.tokenize import sent_tokenize

print("Длина 1 -", len(dvach))
print("Длина 2 -", len(news))

def normalize(text):
    normalized_text = [word.text.strip(punctuation) for word in razdel_tokenize(text)]
    normalized_text = [word.lower() for word in normalized_text if word and len(word) < 20 ]
    return normalized_text

def ngrammer(tokens, n = 2):
    ngrams = []
    for i in range(0,len(tokens)-n+1):
        ngrams.append(' '.join(tokens[i:i+n]))
    return ngrams


def generate(matrix, id2word, word2id, bigram2id, id2bigram, n=100, start = '<start>' '<start>'):
    text = []
    current_idx = word2id[start]

    for i in range(n):

        chosen = np.random.choice(matrix.shape[1], p = matrix[current_idx])
        text.append(id2word[chosen])
        current_bigram = bigram2id[current_idx]
        second_bigram_word = current_bigram.split()[1]

        next_bi = f'{second_bigram_word} {id2word[chosen]}'


        if id2word[chosen] == '<end>':
            chosen = bigram2id[start]
        else:
            current_idx = bigram2id[next_bi]

    return ' '.join(text)


norm_dvach = normalize(dvach)
norm_news = normalize(news)


sentences_dvach = [['<start>', '<start>'] + normalize(text) + ['<end>'] for text in sent_tokenize(dvach)]
sentences_news = [['<start>', '<start>'] + normalize(text) + ['<end>'] for text in sent_tokenize(news)]

print("Длина корпуса токсичных постов в токенах -", len(norm_dvach))
print("Длина корпуса новостных текстов в токенах - ", len(norm_news))


print("Уникальных токенов в корпусе токсичных постов -", len(set(norm_dvach)))
print("Уникальных токенов в корпусе новостных текстов - ", len(set(norm_news)))

vocab_dvach = Counter(norm_dvach)
vocab_news = Counter(norm_news)
vocab_dvach.most_common(10)
vocab_news.most_common(10)

print(vocab_dvach.most_common(10))
print(vocab_news.most_common(10))

probas_dvach = Counter({word:c/len(norm_dvach) for word, c in vocab_dvach.items()})
probas_dvach.most_common(20)

probas_news = Counter({word:c/len(norm_news) for word, c in vocab_news.items()})
probas_news.most_common(20)


unigrams_dvach = Counter()
bigrams_dvach = Counter()
trigrams_dvach = Counter()

for sentence in sentences_dvach:
    unigrams_dvach.update(sentence)
    bigrams_dvach.update(ngrammer(sentence))
    trigrams_dvach.update(ngrammer(sentence, n = 3))


unigrams_news = Counter()
bigrams_news = Counter()
trigrams_news = Counter()

for sentence in sentences_news:
    unigrams_news.update(sentence)
    bigrams_news.update(ngrammer(sentence))
    trigrams_news.update(ngrammer(sentence, 3))

prob = Counter({'news':0, 'dvach':0})


for word in normalize(dvach):
    prob['dvach'] += probas_dvach.get(word, 0)

for word in normalize(news):
    prob['news'] += probas_news.get(word, 0)

print(len(unigrams_dvach))
print(trigrams_news.most_common(10))
print(prob['dvach'])
print(prob['news'])


matrix_dvach = np.zeros((len(bigrams_dvach), len(unigrams_dvach)))
id2word_dvach = list(unigrams_dvach)
word2id_dvach = {word:i for i, word in enumerate(id2word_dvach)}
id2bigram_dvach = list(bigrams_dvach)
bigram2id_dvach = {bigram:i for i, bigram in enumerate(id2bigram_dvach)}

for ngram in trigrams_dvach:
    word1, word2, word3 = ngram.split()
    bigram_key = word1 + ' ' + word3
    print(bigram_key)
    matrix_dvach[bigram2id_dvach[bigram_key]][word2id_dvach[word3]] = trigrams_dvach[ngram]/bigram2id_dvach[bigram_key]


matrix_news = np.zeros((len(bigrams_news), len(unigrams_news)))
id2word_news = list(unigrams_news)
word2id_news = {word:i for i, word in enumerate(id2word_news)}
id2bigram_news = list(bigrams_news)
bigram2id_news = {bigram:i for i, bigram in enumerate(id2bigram_news)}

bigram_key = word1 + ' ' + word2

for ngram in trigrams_news:
     word1, word2, word3 = ngram.split()
     matrix_news[bigram2id_news[bigram_key]][bigram2id_news[word3]] = (trigrams_news[ngram]/bigram2id_news[bigram_key])

print(generate(matrix_dvach, id2word_dvach, word2id_dvach, id2bigram_dvach, bigram2id_dvach).replace('<end>', '\n'))
print(generate(matrix_news, id2word_news, word2id_news, id2bigram_news, bigram2id_news).replace('<end>', '\n'))


#Задание 2
import itertools
from razdel import sentenize
from razdel import tokenize as razdel_tokenize
from pymorphy2 import MorphAnalyzer
from collections import Counter, defaultdict
import numpy as np
import re
from string import punctuation
from nltk.corpus import stopwords
from collections import defaultdict
import gensim
import nltk
from nltk.collocations import *

stops = set(stopwords.words('russian') + ["это", "весь"])
morph = MorphAnalyzer()

def scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    if bigram_count >= min_count:
        corpus_word_count = float(corpus_word_count)
        ca = worda_count / corpus_word_count
        cb = wordb_count / corpus_word_count
        cab = bigram_count / corpus_word_count
        try:
            return np.log(cab / (ca * cb))
        except ValueError:
            return -100000
    else:
        return -100000

    return


def normalize(text):
    tokens = re.findall('[а-яёa-z0-9]+', text.lower())
    normalized_text = [morph.parse(word)[0].normal_form for word \
                       in tokens]
    normalized_text = [word for word in normalized_text if len(word) > 2 and word not in stops]

    return normalized_text

def preprocess(text):
    sents = sentenize(text)
    return [normalize(sent.text) for sent in sents]


corpus = open('C://Users//tonch//CydnieCantFigureOutDisks//lenta//lenta.txt').read()
corpus = preprocess(corpus)

ph = gensim.models.Phrases(corpus, scoring = scorer, threshold = 0)
p = gensim.models.phrases.Phraser(ph)

ph2 = gensim.models.Phrases(p[corpus], scoring = scorer, threshold = 0)
p2 = gensim.models.phrases.Phraser(ph2)

bigram_measures = nltk.collocations.BigramAssocMeasures()
trigram_measures = nltk.collocations.TrigramAssocMeasures()

finder2 = BigramCollocationFinder.from_documents(corpus)

finder3 = TrigramCollocationFinder.from_documents(corpus)

finder2.nbest(bigram_measures.likelihood_ratio, 20)

def scorer(worda_count, wordb_count, bigram_count, len_vocab, min_count, corpus_word_count):
    if bigram_count >= min_count:
        corpus_word_count = float(corpus_word_count)
        ca = worda_count / corpus_word_count
        cb = wordb_count / corpus_word_count
        cab = bigram_count / corpus_word_count
        return np.log(cab / (ca * cb))
    else: 0

#Это более удобнее использовать, потому что это личная функция, мы можем редактировать как нужно к нашим данным.
