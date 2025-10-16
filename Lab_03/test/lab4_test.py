import sys
sys.path.append('Lab_03')

from src.representations.word_embedder import WordEmbedder


print("Loading model...")
model = WordEmbedder('glove-wiki-gigaword-50')
print("Model loaded successfully")

# word = input("\nEnter word: ")
word = "king"
print(f"\nVector of {word}:")
print(model.get_vector(word))

# word1, word2 = [word for word in input("\nEnter word1 and word2: ").split()]
word1, word2 = "king", "queen"
print(f"\nSimilarity of {word1} and {word2}:")
print(model.get_similarity(word1, word2))

word1, word2 = "king", "man"
print(f"\nSimilarity of {word1} and {word2}:")
print(model.get_similarity(word1, word2))

# word = input("\nEnter word: ")
# top_n = int(input("Enter n: "))
word = "computer"
n = 10
print(f"\nTop {n} most similarity of {word}:")
for w, s in model.get_most_similar(word, n):
    print(f"word = {w:20s} score = {s:.4f}")

# doc = input("\nEnter doc: ")
doc = "The queen rules the country."
print(f"\nDoc = {doc}")
doc_vector = model.embed_document(doc)
print(f"Doc vector shape: {doc_vector.shape}")
print(f"Doc vector: {doc_vector}")

model.visualization()