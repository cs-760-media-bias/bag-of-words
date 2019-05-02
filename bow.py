import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('punkt')


def bagOfWords(tweets):
    '''
    [bag_of_words, feature_names] = bagOfWords(tweets)

    Creates BOW feature vector and concatonates with df
    '''

    #bag of words
    count = CountVectorizer()

    bag_of_words = count.fit_transform(tweets)

    #data about bow
    messages_bow = count.transform(tweets)

    print ('\nShape of Sparse Matrix: ', messages_bow.shape)
    print ('Amount of Non-Zero occurences: ', messages_bow.nnz)
    print ('sparsity: %.2f%% \n' % (100.0 * messages_bow.nnz /
                         (messages_bow.shape[0] * messages_bow.shape[1])))

    # Get feature names
    feature_names = count.get_feature_names()

    return bag_of_words, feature_names
