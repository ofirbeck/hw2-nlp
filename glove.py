import re
import nltk
import numpy as np
import matplotlib.pyplot as plt
from nltk.tokenize import word_tokenize
from sklearn.decomposition import PCA
from collections import defaultdict

# Ensure nltk dependencies are downloaded
nltk.download("punkt")


# Step 1: Preprocess the text data
def preprocess_text(text):

    # Clean and tokenize the text
    text = re.sub(r"[^\w\s]", "", text.lower())  # Remove punctuation and lowercase
    sentences = nltk.sent_tokenize(text)  # Tokenize into sentences
    tokenized_sentences = [
        word_tokenize(sentence) for sentence in sentences
    ]  # Tokenize each sentence into words

    return tokenized_sentences


# Step 2: Train GloVe model (using Gensim's Word2Vec as a proxy for GloVe model)
class GloVe:
    def __init__(self, corpus, vector_size, window, min_count, learning_rate, epochs):
        self.corpus = corpus
        self.vector_size = vector_size
        self.window = window
        self.min_count = min_count
        self.learning_rate = learning_rate
        self.epochs = epochs

        # Build vocabulary and co-occurrence matrix
        self.build_vocab()
        self.build_cooccurrence_matrix()

    def build_vocab(self):
        self.vocab = defaultdict(int)
        for sentence in self.corpus:
            for word in sentence:
                self.vocab[word] += 1
        # Filter words that appear less than min_count
        self.vocab = {
            word: count for word, count in self.vocab.items() if count >= self.min_count
        }
        self.word_to_index = {word: idx for idx, word in enumerate(self.vocab.keys())}
        self.index_to_word = {idx: word for word, idx in self.word_to_index.items()}

    def build_cooccurrence_matrix(self):
        self.cooccurrence_matrix = np.zeros((len(self.vocab), len(self.vocab)))
        for sentence in self.corpus:
            for i, word in enumerate(sentence):
                word_idx = self.word_to_index.get(word)
                for j in range(
                    max(i - self.window, 0), min(i + self.window, len(sentence))
                ):
                    context_word = sentence[j]
                    if word != context_word:
                        context_idx = self.word_to_index.get(context_word)
                        self.cooccurrence_matrix[word_idx][context_idx] += 1

    def train(self):
        self.word_vectors = np.random.rand(len(self.vocab), self.vector_size)
        for epoch in range(self.epochs):
            for i in range(len(self.vocab)):
                for j in range(len(self.vocab)):
                    if self.cooccurrence_matrix[i][j] > 0:
                        cooccurrence = self.cooccurrence_matrix[i][j]
                        prediction = np.dot(self.word_vectors[i], self.word_vectors[j])
                        gradient = (
                            2 * (prediction - cooccurrence) * self.word_vectors[j]
                        )
                        self.word_vectors[i] -= self.learning_rate * gradient
            print(f"Epoch {epoch + 1}/{self.epochs} completed.")

    def get_word_vector(self, word):
        return self.word_vectors[self.word_to_index[word]]


# Step 4: PCA Visualization
def visualize_embeddings_with_pca(model, words):
    word_vectors = [model.get_word_vector(word) for word in words]

    # Apply PCA for dimensionality reduction
    pca = PCA(n_components=2)
    reduced_vectors = pca.fit_transform(word_vectors)

    plt.figure(figsize=(10, 8))
    plt.scatter(reduced_vectors[:, 0], reduced_vectors[:, 1])

    for i, word in enumerate(words):
        plt.annotate(word, (reduced_vectors[i, 0], reduced_vectors[i, 1]))

    plt.title("PCA of GloVe Word Embeddings (2D)")
    plt.show()