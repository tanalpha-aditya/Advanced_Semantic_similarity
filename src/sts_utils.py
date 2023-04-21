import pickle
from scipy import spatial
from scipy.stats import pearsonr

def load_file(file_path):
    with open(file_path, 'rb') as handle:
        file = pickle.load(handle)
    return file

def sts_score(sim_score):
    sts_score = (sim_score+1) * 2.5
    return sts_score

def get_sts_scores(emb1, emb2):
    sim_score = 1 - spatial.distance.cosine(emb1, emb2)
    return sim_score

def pearson_corr(y_true, y_pred):
    """
    Calculate Pearson correlation coefficient between two arrays.
    """
    corr, _ = pearsonr(y_true, y_pred)
    return corr

