import re
import nltk
import pickle
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download('stopwords')
nltk.download('wordnet')

lemmatizer = WordNetLemmatizer()
stop_words_es = set(stopwords.words('spanish'))
stop_words = set(stopwords.words('english'))

def preprocess_text_en(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Replace numbers with num
    text = re.sub(r'\d+', '', text)
    # Lower case
    text = text.lower()
    # sent_token = text.split()
    sent_token = nltk.word_tokenize(text, language='english')
    # stop words removal
    sent_token = [word for word in sent_token if word.lower() not in stop_words]
    # Lemmatize
    sent_token = [lemmatizer.lemmatize(word) for word in sent_token]
    # Stemming
    # sent_token = [ps.stem(word) for word in sent_token]
    return sent_token

def preprocess_text_es(text):
    # Remove punctuation
    text = re.sub(r'[^\w\s]', '', text)
    # Replace numbers with num
    text = re.sub(r'\d+', '', text)
    # Lower case
    text= text.lower()
    #sent_token = text.split()
    sent_token = nltk.word_tokenize(text, language='spanish')
    #stop words removal
    sent_token = [word for word in sent_token if word.lower() not in stop_words_es]
    # Lemmatize
    sent_token = [lemmatizer.lemmatize(word) for word in sent_token]
    # Stemming
    #sent_token = [ps.stem(word) for word in sent_token]
    return sent_token

def load_dict(lang):
    if lang == "en":
        with open('../data/word_dict_en_v1.pickle', 'rb') as f:
            vocab_dict = pickle.load(f)
    elif lang == "es":
        with open('../data/word_dict_es_v1.pickle', 'rb') as f:
            vocab_dict = pickle.load(f)
    return vocab_dict


class MyDataset(Dataset):
    def __init__(self, sentences1, sentences2, word_to_ix_en, word_to_ix_es ):
        self.sentences1 = sentences1
        self.sentences2 = sentences2
        self.word_to_ix_en = word_to_ix_en
        self.word_to_ix_es = word_to_ix_es

    def __len__(self):
        return max(len(self.sentences1),len(self.sentences2))

    def __getitem__(self, idx):
        unk_token1 = self.word_to_ix_en['unk']
        unk_token2 = self.word_to_ix_es['unk']
        sentence1 = self.sentences1[idx]
        sentence2 = self.sentences2[idx]
        seq1 = [self.word_to_ix_en[word] if word in self.word_to_ix_en else unk_token1 for word in sentence1]
        seq2 = [self.word_to_ix_es[word] if word in self.word_to_ix_es else unk_token2 for word in sentence2]
        return seq1, seq2

    def collate_fn(self, batch):
        sequences1, sequences2 = zip(*batch)
        padded_seqs1 = pad_sequence([torch.LongTensor(seq) for seq in sequences1], batch_first=True, padding_value=0)
        padded_seqs2 = pad_sequence([torch.LongTensor(seq) for seq in sequences2], batch_first=True, padding_value=0)
        #return padded_seqs1, padded_seqs2, torch.tensor(scores, dtype=torch.float)
        return padded_seqs1, padded_seqs2



class SiameseBiLSTM(nn.Module):
    def __init__(self, hidden_size, num_layers, embedding_dim, embd_matrix1, embd_matrix2, dropout=0.2):
        super(SiameseBiLSTM, self).__init__()

        # LSTM parameters
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.embedding_dim = embedding_dim
        self.embd_matrix1 = embd_matrix1
        self.embd_matrix2 = embd_matrix2

        # Word embeddings
        self.word_embeddings1 = nn.Embedding(len(self.embd_matrix1), embedding_dim)
        self.word_embeddings1.weight = nn.Parameter(torch.from_numpy(self.embd_matrix1))
        self.word_embeddings1.weight.requires_grad = False

        self.word_embeddings2 = nn.Embedding(len(self.embd_matrix2), embedding_dim)
        self.word_embeddings2.weight = nn.Parameter(torch.from_numpy(self.embd_matrix2))
        self.word_embeddings2.weight.requires_grad = False

        # BiLSTM layers
        self.bilstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_size, num_layers=num_layers,
                              batch_first=True, bidirectional=True)

        # Dropout layer
        self.dropout = nn.Dropout(dropout)

        # Attention layers
        self.attention_fc = nn.Linear(hidden_size * 2, 1)
        self.attention_softmax = nn.Softmax(dim=1)

        # Similarity scoring layer
        self.fc = nn.Linear(hidden_size * 4,
                            1)  # 4 because we concatenate forward and backward hidden states of both LSTMs

    def forward_once_en(self, sentence):
        # Word embeddings
        embeds = self.word_embeddings1(sentence)

        # BiLSTM
        lstm_out, _ = self.bilstm(embeds)

        # Apply dropout to hidden layers
        lstm_out = self.dropout(lstm_out)

        # Attention mechanism
        attention_weights = self.attention_softmax(self.attention_fc(lstm_out))
        lstm_out = lstm_out * attention_weights
        lstm_out = lstm_out.sum(dim=1)

        return lstm_out

    def forward_once_es(self, sentence):
        # Word embeddings
        embeds = self.word_embeddings2(sentence)

        # BiLSTM
        lstm_out, _ = self.bilstm(embeds)

        # Apply dropout to hidden layers
        lstm_out = self.dropout(lstm_out)

        # Attention mechanism
        attention_weights = self.attention_softmax(self.attention_fc(lstm_out))
        lstm_out = lstm_out * attention_weights
        lstm_out = lstm_out.sum(dim=1)

        return lstm_out

    def forward(self, sentence1, sentence2):
        # Process sentence 1
        output1 = self.forward_once_en(sentence1)

        # Process sentence 2
        output2 = self.forward_once_es(sentence2)

        # Concatenate outputs of both LSTMs
        concatenated = torch.cat((output1, output2), dim=1)

        # Pass through similarity scoring layer
        similarity_score = torch.sigmoid(self.fc(concatenated))

        return similarity_score

def load_model():
    model = torch.load("../data/cross_siamese_model_v1.pt")
    return model

def run_prediction(dataloader):
    model = load_model()
    model.eval()
    with torch.no_grad():
        for k, (test_sentence1, test_sentence2) in enumerate(dataloader):
            test_sentence1_tensor = test_sentence1
            test_sentence2_tensor = test_sentence2
            test_output = model(test_sentence1_tensor, test_sentence2_tensor)*5.0  #multiply by 5.0 because the model fc is sigmoid layer
            return test_output.tolist()[0][0]

def generate_similarity_score(sent1, sent2):
    sent1_tokens = list(preprocess_text_en(sent1))
    sent2_tokens = list(preprocess_text_es(sent2))
    word_to_ix_en = load_dict("en")
    word_to_ix_es = load_dict("es")
    prediction_dataset = MyDataset(sent1_tokens, sent2_tokens, word_to_ix_en, word_to_ix_es )
    prediction_dataloader = DataLoader(prediction_dataset, batch_size=1, shuffle=True, collate_fn=prediction_dataset.collate_fn)
    similarity_score = run_prediction(prediction_dataloader)
    return similarity_score

if __name__ == "__main__":
    """Sample input sentences to try:
        input1: A man is playing a flute.
        input2: Un hombre está tocando una flauta de bambú.
        """
    input1 = input("Enter your english sentence:\t")
    input2 = input("Enter your spanish sentence:\t")
    print("Generating the cross lingual similarity score......")
    print("The predicted semantic similarity score is ", generate_similarity_score(input1, input2))
