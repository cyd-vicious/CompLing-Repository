import urllib.request
import string
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
import regex
from gensim.utils import tokenize
from gensim.summarization.textcleaner import split_sentences
from rusenttokenize import ru_sent_tokenize
from razdel import sentenize
from razdel import tokenize as razdel_tokenize
from nltk import sent_tokenize
from nltk.tokenize import word_tokenize, wordpunct_tokenize
from pymorphy2 import MorphAnalyzer
from pymystem3 import Mystem
from nltk.corpus import stopwords
from string import punctuation
import re, os, json
mystem = Mystem()
morph = MorphAnalyzer()
from nltk.stem.snowball import SnowballStemmer
mystem = Mystem()
morph = MorphAnalyzer()
stemmer = SnowballStemmer('russian')
url = "https://raw.githubusercontent.com/mannefedov/compling_nlp_hse_course/master/data/zhivago.txt"
file = urllib.request.urlopen(url)
text = file.read().decode('utf-8')
f = open("C://Users//tonch//Desktop//compling_hw1.txt", "w", encoding = "utf-8")

def remove_tags_1(text):
    return re.sub(r'<[^>]+>', '', text)

def findDuplicates(sentences):
    seen = {}
    dupes = []
    for x in sentences:
        if x not in seen:
            seen[x] = 1
        else:
            if seen[x] == 1:
                dupes.append(x)
            seen[x] += 1
    return dupes

def most_frequent(List):
    filteredList = []
    for word in List:
        if len(word) > 6:
            filteredList.append(word)
    return max(set(filteredList), key = filteredList.count)

def potentialStems(wordList, f):
    counter = 0
    for i, word in enumerate(wordList):
        if counter == 50:
            f.write("Найдено 50 потенциальных ошибок, заканчиваю обработку")
            f.write("\n")
            break
        print(i)
        print("/")
        print(len(wordList))
        if len(word) < 2:
            continue
        wordstem = stemmer.stem(word)
        if len(wordstem) < 2:
            continue
        searchRule = word + "{i<=10}"
        matches = []
        for potentialMatch in wordList:
            if regex.fullmatch(searchRule, potentialMatch):
                if potentialMatch != word:
                    matches.append(potentialMatch)
        if len(matches) > 1:
            f.write("Слово: " + word)
            f.write("\n")
            f.write("Стемм: " + wordstem)
            f.write("\n")
            f.write("Потенциальные ошибки: ")
            f.write("\n")
            cleanMatches = set(matches)
            for curr in cleanMatches:
                f.write(curr)
                f.write("\n")
            counter += 1

text = remove_tags_1(text)
punctuation = string.punctuation
text = text.lower()
sentences = ru_sent_tokenize(text)
print("1//5!!!!!!!!!!!!!!!!!!")
f.write("Предложения: ")
f.write("/n")
f.writelines(sentences)
f.write("\n")
[word.strip(string.punctuation) for word in sentences]
words = word_tokenize(text)
f.write("Слова: ")
f.write("\n")
listofwords = []
for current in words:
    listofwords.append(current)
    f.write(current)
    f.write("\n")
duplicates = findDuplicates(sentences)
print("2//5!!!!!!!!!!!!!!!!!!!!")
f.write("Потворяющиеся предложения: ")
f.write("\n")
f.writelines(duplicates)
f.write("\n")
f.write("Самое частое слово: ")
f.write("\n")
f.write(most_frequent(listofwords))
f.write("\n")
stemList = []
print("3//5!!!!!!!!!!!!!!!!!!!!")
f.write("Список стемминга: ")
f.write("\n")
wrongStemming = []
for word in words:
    stemList.append(stemmer.stem(word))
    if stemmer.stem(word) == word and len(word) > 4:
        wrongStemming.append(stemmer.stem(word))
        f.write(word)
        f.write("\n")
        f.write(stemmer.stem(word))
    if len(wrongStemming) == 0:
        f.write("Нет результатов")
        f.write("\n")
f.write("Некорректный стемминг: ")
f.write("\n")
#f.writelines(wrongStemming)
f.write("\n")
print("4//5!!!!!!!!!!!!!!!!!!!!")
stops = stopwords.words('russian')
f.writelines(stops)
f.write("\n")

potentialStems(set(words), f)

words_analyzed = [morph.parse(token) for token in word_tokenize(text)]
f.write("Лемматизация токенов: ")
f.write("\n")
for x in words_analyzed:
    f.write(str(x))
    f.write("\n")
mystem.lemmatize(text)[:10]
lemmas = mystem.lemmatize(text)
print("5//5!!!!!!!!!!!!!!!!!!!!")
f.write("Лемматизация текста: ")
f.write("\n")
for x in lemmas:
    f.write(x)
    f.write("\n")
    print("Found")
f.write("\n")
f.close()
print("Done")

#есть ли в тексте повторяющиеся корректные предложения?
#ответ: парило. странно. он открыл глаза. толпа росла. да.

#какой самый частотный токен в тексте длиннее 6 символов?
#ответ: андреевич

#два разных слова ошибочно свелись к одинаковой основе
#ответ:
# Слово: перестав
# Стемм: переста
# Потенциальные ошибки:
# переставлял
# переставшие
# перестановок
# переставал
# переставая
# переставали
# переставших
# переставил
#
# Слово: наук
# Стемм: наук
# Потенциальные ошибки:
# науке
# наутек
# науками
# науки
# наушниками
# наукою
# наукам
# наука

#слово не изменилось после стемминга (слово должно быть русским и длиннее 4 символов)
#ответ: я не поняла как это сделать, я пыталалсь много раз и не получилось.

#Какие ещё слова вы бы туда добавили?
#ответ: "мол", "таки", "неужто", "некой", "одной"
#несколько из этих слов в старом стиле языка, которые не находятся в наборе stopwords. последнее слово я заметила что есть похожи в
#оригинальном списке, но не именно это. думала, что жто странно, и нужно добавить.

#Что в данном случае лучше для лемматизации mystem или pymorphy?
#ответ: pymorphy, потому что он быстрее обрабатывается

#извините, что так поздно сдала домашку, у меня было проблема с pip, а я еще не очень хорошо знаю питон (я сейчас слушаю базовый курс)