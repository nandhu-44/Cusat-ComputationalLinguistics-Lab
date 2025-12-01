import numpy as np
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from gensim.models import Word2Vec


def create_corpus():
    """Create sample corpus for word embeddings"""
    return [
        "machine learning is a subset of artificial intelligence",
        "deep learning is a type of machine learning",
        "neural networks are used in deep learning",
        "python is popular for machine learning",
        "data science involves statistics and programming",
        "natural language processing is part of artificial intelligence",
        "computer vision helps machines understand images",
        "reinforcement learning trains agents through rewards",
        "supervised learning uses labeled data",
        "unsupervised learning finds patterns in unlabeled data",
        "the cat sat on the mat",
        "dogs are loyal pets",
        "birds can fly high in the sky",
        "fish swim in water",
        "trees provide oxygen and shade"
    ]


def train_word2vec(sentences, vector_size=50, window=3, min_count=1):
    """Train Word2Vec model"""
    tokenized = [sent.lower().split() for sent in sentences]
    model = Word2Vec(sentences=tokenized, vector_size=vector_size, window=window, 
                     min_count=min_count, workers=4, epochs=100)
    return model


def cluster_and_plot(model, method='tsne'):
    """Cluster words and visualize using PCA or t-SNE"""
    words = list(model.wv.index_to_key)
    vectors = np.array([model.wv[word] for word in words])
    
    if method == 'pca':
        reducer = PCA(n_components=2)
        coords = reducer.fit_transform(vectors)
        title = 'Word Clustering using PCA'
    else:
        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(words)-1))
        coords = reducer.fit_transform(vectors)
        title = 'Word Clustering using t-SNE'
    
    plt.figure(figsize=(14, 10))
    plt.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=100)
    
    for i, word in enumerate(words):
        plt.annotate(word, (coords[i, 0], coords[i, 1]), 
                    fontsize=9, alpha=0.8)
    
    plt.title(title, fontsize=14)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    filename = f'word_clusters_{method}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"Saved plot: {filename}")
    plt.show()


def find_similar_words(model, word, topn=5):
    """Find most similar words"""
    if word in model.wv:
        similar = model.wv.most_similar(word, topn=topn)
        return similar
    return []


def main():
    corpus = create_corpus()
    
    model = train_word2vec(corpus, vector_size=50, window=3)
    
    vocab_size = len(model.wv)
    print(f"Vocabulary size: {vocab_size} words")
    print(f"Vector dimensions: {model.wv.vector_size}")
    
    print("\nWord Similarities:")
    
    test_words = ['learning', 'machine', 'cat', 'python', 'data']
    
    for word in test_words:
        similar = find_similar_words(model, word, topn=5)
        if similar:
            print(f"\n'{word}' is similar to:")
            for sim_word, score in similar:
                print(f"  {sim_word}: {score:.3f}")
    
    print("\nVector Space Representation:")
    
    sample_words = ['machine', 'learning', 'deep', 'python']
    for word in sample_words:
        if word in model.wv:
            vec = model.wv[word]
            print(f"{word}: [{vec[0]:.3f}, {vec[1]:.3f}, {vec[2]:.3f}, ...]")
    
    print("\nGenerating Clusters and Plots:")
    cluster_and_plot(model, method='pca')    
    cluster_and_plot(model, method='tsne')

if __name__ == "__main__":
    main()
