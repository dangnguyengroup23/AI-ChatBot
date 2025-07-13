import nltk
import numpy as np
# nltk.download('punkt')
from nltk.stem.porter import PorterStemmer
stemmer = PorterStemmer()




def tokenize(sentence):
    return nltk.word_tokenize(sentence)

def stem(word):
    return stemmer.stem(word.lower())

def bagOfWord(tokenizedSentence,all_words):
    tokenizedSentence = [stem(w) for w in tokenizedSentence]
    bag = np.zeros(len(all_words),dtype=np.float32)

    for idx,w in enumerate(all_words):
        if w in tokenizedSentence:
            bag[idx]=1.0
    return bag

sentene = ['hello','how','are','you']
a = ['hi','hello','i','you']

bag = bagOfWord(sentene,a)
bag = bag.reshape(1,bag.shape[0])
print(bag.shape)