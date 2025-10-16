import re
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lower, regexp_replace, split
from pyspark.ml.feature import Word2Vec
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np

def main():
    # 1️. Initialize Spark Session
    spark = (
        SparkSession.builder
        .appName("Spark Word2Vec Demo")
        .master("local[*]")
        .getOrCreate()
    )
    spark.sparkContext.setLogLevel("WARN")

    # 2️. Load dataset (C4 JSON)
    data_path = "Lab_02/data/c4-train.00000-of-01024-30K.json.gz"
    df = spark.read.json(data_path)
    df = df.select("text").na.drop(subset=["text"])

    print(f"Loaded {df.count()} documents")

    # 3️. Preprocessing
    df_clean = (
        df.withColumn("text", lower(col("text")))
          .withColumn("text", regexp_replace(col("text"), r"[^a-z\s]", " "))
          .withColumn("tokens", split(col("text"), r"\s+"))
    )

    # 4️. Configure & Train Word2Vec
    word2Vec = Word2Vec(
        vectorSize=100,
        minCount=2,
        inputCol="tokens",
        outputCol="result"
    )

    print("Training Word2Vec model...")
    model = word2Vec.fit(df_clean)
    print("Training completed.")

    # 5️. Demonstrate the model
    word = "computer"
    try:
        synonyms = model.findSynonymsArray(word, 5)
        print(f"\nTop 5 words similar to '{word}':")
        for synonym, sim in synonyms:
            print(f"   {synonym:15s}  ->  similarity = {sim:.4f}")
    except:
        print(f"Could not find synonyms for '{word}'")


    vectors_df = model.getVectors()
    vectors_data = vectors_df.collect()

    words = [row["word"] for row in vectors_data]
    vectors = np.array([row["vector"].toArray() for row in vectors_data])

    pca = PCA(n_components=2)
    vecs = pca.fit_transform(vectors)

    plt.figure(figsize=(8, 7))
    for i, word in enumerate(words[:100]):
        plt.scatter(vecs[i, 0], vecs[i, 1], s=15, alpha=0.7)
        plt.annotate(word, xy=(vecs[i, 0], vecs[i, 1]), fontsize=8)

    plt.title("Word Embeddings 2D Visualization (Spark Word2Vec)")
    plt.show()


    # 6️. Stop Spark session
    spark.stop()
    print("Spark session stopped.")

if __name__ == "__main__":
    main()
