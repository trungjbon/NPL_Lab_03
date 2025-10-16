import sys
sys.path.append('Lab_01')

from src.preprocessing.regex_tokenizer import RegexTokenizer
from gensim.downloader import load
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class WordEmbedder:
    def __init__(self, model_name: str):
        self.model = load(model_name)
        self.vector_size = self.model.vector_size
        self.words = list(self.model.key_to_index.keys())
        self.vectors = [self.model[word] for word in self.words]

    def get_vector(self, word: str):
        try:
            return self.model[word]
        except:
            print(f"Word '{word}' not in the vocabulary")
            return None

    def get_similarity(self, word1: str, word2: str):
        try:
            return self.model.similarity(word1, word2)
        except:
            print(f"One or both words ('{word1}', '{word2}') are not in the vocabulary.")
            return None
    
    def get_most_similar(self, word: str, top_n: int = 10):
        try:
            return self.model.most_similar(word, topn=top_n)
        except:
            print(f"Word '{word}' not in vocabulary.")
            return None
        
    def embed_document(self, document: str):
        tokenizer = RegexTokenizer()
        tokens = tokenizer.tokenize(document)
        
        vectors = [self.get_vector(word) for word in tokens if word in self.words]

        if (not vectors):
            return np.zeros(self.vector_size)
        
        return np.mean(vectors, axis=0)
    
    def visualization(self):
        pca = PCA(n_components=2)
        vecs = pca.fit_transform(self.vectors)

        plt.figure(figsize=(8, 7))

        for i, word in enumerate(self.words[:100]):
            plt.scatter(vecs[i, 0], vecs[i, 1], s=15, alpha=0.7)
            plt.annotate(word, xy=(vecs[i, 0], vecs[i, 1]), fontsize=8)

        plt.title("Word Embeddings 2D Visualization")
        plt.show()
