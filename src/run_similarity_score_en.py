import numpy as np
import re
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
from gensim.models import KeyedVectors
from sts_utils import load_file, sts_score, get_sts_scores


class STS_Score:
    def __init__(self):
        self.lemmatizer = WordNetLemmatizer()
        self.ps = PorterStemmer()
        self.word_dict = load_file("../data/word_dict.pickle")
        model_path = "../data/GoogleNews-vectors-negative300.bin"
        self.model = KeyedVectors.load_word2vec_format(model_path, binary=True)

    def preprocess_text(self, text):
        # Remove punctuation
        text = re.sub(r'[^\w\s]', '', text)
        # Replace numbers with num
        text = re.sub(r'\d+', '', text)
        # Lower case
        text = text.lower()
        sent_token = text.split()
        # Lemmatize
        sent_token = [self.lemmatizer.lemmatize(word) for word in sent_token]
        # Stemming
        sent_token = [self.ps.stem(word) for word in sent_token]
        return sent_token

    def unk_replace(self, word, word_dict):
        if word not in word_dict:
            return "unk"
        else:
            if word_dict[word] < 2:
                return "unk"
        return word

    # define a function to generate sentence embeddings
    def get_sentence_embedding(self, sentence, max_length=30):
        words = sentence
        # filter out words that are not present in the model's vocabulary
        words = [word for word in words if word in self.model.key_to_index]
        # generate word embeddings for each word
        embeddings = [self.model[word] for word in words]
        #mean strategy
        embedding = np.mean(embeddings, axis=0)
        return embedding

    def generate_similarity_score(self, sent1, sent2):
        sent1_token = self.preprocess_text(sent1)
        sent2_token = self.preprocess_text(sent2)
        sent1_unk_token = [self.unk_replace(word, self.word_dict) for word in sent1_token]
        sent2_unk_token = [self.unk_replace(word, self.word_dict) for word in sent2_token]
        sent1_embedding = self.get_sentence_embedding(sent1_unk_token)
        sent2_embedding = self.get_sentence_embedding(sent2_unk_token)
        normalized_cos_scores = sts_score(get_sts_scores(sent1_embedding, sent2_embedding))
        return normalized_cos_scores


if __name__ == "__main__":
    """Sample input sentences to try:
    input1: A man is cutting up a cucumber.
    input2: A man is slicing a cucumber.
    """
    input1 = input("Enter your first sentence:\t")
    input2 = input("Enter your second sentence:\t")
    print("Generating the similarity score......")
    sts = STS_Score()
    print("The semantic similarity score is", sts.generate_similarity_score(input1, input2))
    print("The explainability can be generated through sts_explainability module ")
