"""
Reduce 384-dimensional embeddings to 2D and show clusters.
Uses PCA for dimensionality reduction.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from rag.embeddings import EmbeddingModel
from sklearn.decomposition import PCA
import numpy as np


def main():
    model = EmbeddingModel()
    model.load()
    
    # Three clusters of sentences
    tech = [
        "Python is a programming language",
        "JavaScript runs in the browser",
        "Machine learning uses neural networks",
        "The API returns a JSON response",
        "Git is used for version control",
    ]
    food = [
        "I love eating pizza",
        "The pasta was delicious",
        "She baked a chocolate cake",
        "We ordered sushi for dinner",
        "The recipe needs fresh basil",
    ]
    sports = [
        "The team won the championship",
        "He scored a goal in the last minute",
        "She ran the marathon in under 3 hours",
        "The tennis match went to five sets",
        "They practiced basketball all morning",
    ]
    
    all_texts = tech + food + sports
    labels = ["tech"] * 5 + ["food"] * 5 + ["sports"] * 5
    
    # Embed all sentences
    vectors = model.embed_batch(all_texts)
    
    # Reduce to 2D
    pca = PCA(n_components=2)
    coords = pca.fit_transform(vectors)
    
    # Print clusters
    print("=" * 60)
    print("  EMBEDDING SPACE VISUALIZATION (2D projection)")
    print("  Sentences with similar meaning cluster together")
    print("=" * 60)
    
    symbols = {"tech": "◆", "food": "●", "sports": "▲"}
    
    for label_type in ["tech", "food", "sports"]:
        print(f"\n  {symbols[label_type]} {label_type.upper()}")
        for i, (text, lbl) in enumerate(zip(all_texts, labels)):
            if lbl == label_type:
                x, y = coords[i]
                print(f"    ({x:>6.2f}, {y:>6.2f})  {text[:45]}")
    
    # Show inter-cluster vs intra-cluster distances
    print(f"\n{'─' * 60}")
    print("  Average cosine similarity WITHIN clusters:")
    for cluster_name, cluster_texts in [("tech", tech), ("food", food), ("sports", sports)]:
        vecs = model.embed_batch(cluster_texts)
        sims = []
        for i in range(len(vecs)):
            for j in range(i + 1, len(vecs)):
                sims.append(float(np.dot(vecs[i], vecs[j])))
        print(f"    {cluster_name:>8s}: {np.mean(sims):.3f}")
    
    print("\n  Average cosine similarity BETWEEN clusters:")
    for name_a, texts_a, name_b, texts_b in [
        ("tech", tech, "food", food),
        ("tech", tech, "sports", sports),
        ("food", food, "sports", sports),
    ]:
        vecs_a = model.embed_batch(texts_a)
        vecs_b = model.embed_batch(texts_b)
        sims = []
        for va in vecs_a:
            for vb in vecs_b:
                sims.append(float(np.dot(va, vb)))
        print(f"    {name_a:>8s} ↔ {name_b:<8s}: {np.mean(sims):.3f}")
    
    print(f"\n  📌 Within-cluster similarity should be MUCH higher")
    print(f"     than between-cluster similarity.")
    print(f"     This is what makes semantic search work —")
    print(f"     your query lands near the relevant cluster.")
    print("=" * 60)


if __name__ == "__main__":
    main()