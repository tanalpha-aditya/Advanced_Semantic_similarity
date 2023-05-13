from sentence_transformers import SentenceTransformer, util
from transformers import BertTokenizer
import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler

# Convert sentences to input features
def convert_sentences_to_features(sentences, tokenizer, max_seq_length):
    input_ids = []
    attention_masks = []
    token_type_ids = []

    for sent in sentences:
        encoded_dict = tokenizer.encode_plus(
                            sent[0],                      # Sentence 1
                            sent[1],                      # Sentence 2
                            add_special_tokens = True,    # Add '[CLS]' and '[SEP]'
                            max_length = max_seq_length,  # Pad or truncate sentences
                            truncation=True,
                            #pad_to_max_length = True,
                            padding = 'max_length',
                            return_attention_mask = True, # Construct attention masks
                            return_tensors = 'pt',        # Return PyTorch tensors
                            truncation_strategy='longest_first'
                       )

        # Add the encoded sentence to the list
        input_ids.append(encoded_dict['input_ids'])
        attention_masks.append(encoded_dict['attention_mask'])
        token_type_ids.append(encoded_dict['token_type_ids'])

    # Convert the lists to tensors
    input_ids = torch.cat(input_ids, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    token_type_ids = torch.cat(token_type_ids, dim=0)

    return input_ids, attention_masks, token_type_ids

class SBertModel():
    def __init__(self, wanted_model='stsb-roberta-large'):
        self.sts_model = SentenceTransformer(wanted_model)

    def __call__(self, sent_pairs):
        all_sents = []
        for pair in sent_pairs:
            all_sents += [pair[0],pair[1]]

        embds = self.sts_model.encode(all_sents)
        scores = []
        for i in range(int(len(all_sents)/2)):
            scores.append(float(util.pytorch_cos_sim(embds[i*2], embds[i*2+1])[0][0]))

        return np.array(scores)

class SBertModel_Custom():
    def __init__(self, wanted_model_path='semantic_sim_bert_v3.pt'):
        self.sts_model = torch.load(wanted_model_path)
        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def __call__(self, sent_pairs):
        all_sents = []
        labels = []
        for pair in sent_pairs:
            all_sents.append([pair[0],pair[1]])
            labels.append(0) #This is redundant as we are only predicting


        input_ids, attention_masks, token_type_ids = convert_sentences_to_features(all_sents, self.tokenizer, 100)
        labels = torch.tensor(labels, dtype=torch.float)
        data = TensorDataset(input_ids, attention_masks, token_type_ids, labels)
        dataloader = DataLoader(data, batch_size=1)
        self.sts_model.eval()
        y_pred_scores = []
        with torch.no_grad():
            for step, batch in enumerate(dataloader):
                batch_input_ids, batch_attention_masks, _, batch_labels = tuple(t for t in batch)
                outputs = self.sts_model(batch_input_ids, token_type_ids=None, attention_mask=batch_attention_masks,
                                    labels=batch_labels)
                list_array = outputs[1].tolist()
                y_pred_scores.extend([i[0] for i in list_array])
        return np.array(y_pred_scores)



        embds = self.sts_model.encode(all_sents)
        scores = []
        for i in range(int(len(all_sents)/2)):
            scores.append(float(util.pytorch_cos_sim(embds[i*2], embds[i*2+1])[0][0]))

        return np.array(scores)
