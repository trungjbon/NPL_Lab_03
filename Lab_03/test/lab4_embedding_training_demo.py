import os
from gensim.models import Word2Vec
from gensim.utils import simple_preprocess
from gensim.models.callbacks import CallbackAny2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

class EpochLogger(CallbackAny2Vec):
    def __init__(self):
        self.epoch = 0
        self.loss_previous_step = 0

    def on_epoch_begin(self, model):
        print(f"\nStarting epoch {self.epoch + 1}...")

    def on_epoch_end(self, model):
        loss_now = model.get_latest_training_loss()
        loss_epoch = loss_now - self.loss_previous_step
        self.loss_previous_step = loss_now
        tokens = model.corpus_total_words
        avg_loss = loss_epoch / tokens

        print(f"Finished epoch {self.epoch + 1}")
        print(f"Average loss per token: {avg_loss:.6f}")
        
        self.epoch += 1


# 1. STREAM DATA
class SentenceStreamer:
    def __init__(self, file_path):
        self.file_path = file_path

    def __iter__(self):
        with open(self.file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                yield simple_preprocess(line)  # tokenize, lowercase, remove punctuation


# 2. TRAIN MODEL
def train_word2vec(data_path: str, save_path: str):
    print(f"Reading data from: {data_path}")
    sentences = SentenceStreamer(data_path)

    epoch_logger = EpochLogger()

    print("Training Word2Vec model...")
    model = Word2Vec(
        sentences=sentences,
        vector_size=100,     # embedding size
        window=5,            # context window
        min_count=2,         # ignore rare words
        workers=4,           # use multi-core
        sg=1,                # skip-gram (sg=1) or CBOW (sg=0)
        epochs=10,
        compute_loss=True,
        callbacks=[epoch_logger]
    )

    # 3. SAVE MODEL
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    model.save(save_path)
    print(f"\nModel saved to {save_path}")


# 4. DEMONSTRATE USAGE
def demo_model(model):
    word = "king"
    n = 5
    print(f"\nTop {n} most similarity of {word}:")
    try:
        for w, s in model.wv.most_similar(word, topn=5):
            print(f"{w:20s}: {s:.4f}")
    except:
        print(f"Word {word} not in vocabulary.")

    w1 = "woman"
    w2 = "king"
    w3 = "man"
    print(f"\nAnalogy vec({w1}) + vec({w2}) − vec({w3}):")
    try:
        result = model.wv.most_similar(positive=['woman', 'king'], negative=['man'], topn=1)
        print(result)
    except:
        print("Analogy words not in vocabulary.")


    words = list(model.wv.key_to_index.keys())
    vectors = [model.wv[word] for word in words]
    pca = PCA(n_components=2)
    vecs = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 7))

    for i, word in enumerate(words[:100]):
        plt.scatter(vecs[i, 0], vecs[i, 1], s=15, alpha=0.7)
        plt.annotate(word, xy=(vecs[i, 0], vecs[i, 1]), fontsize=8)

    plt.title("Word Embeddings 2D Visualization")
    plt.show()


if __name__ == "__main__":
    data_file = "Lab_01/data/UD_English-EWT/en_ewt-ud-train.txt"
    save_file = "Lab_03/results/word2vec_ewt.model"

    train_word2vec(data_file, save_file)
    model = Word2Vec.load("Lab_03/results/word2vec_ewt.model")
    demo_model(model)
