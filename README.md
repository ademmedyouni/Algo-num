# Numerical Matrix Analysis: Document Retrieval Systems

## Project Overview
This project explores the application of matrix analysis techniques, particularly Singular Value Decomposition (SVD), to improve document retrieval systems. It includes the implementation of vector space modeling, query-based search, and dimensionality reduction techniques to enhance search accuracy and efficiency.

## Key Features
- **Vector Space Modeling**: Represents documents and queries as vectors in a term-document matrix.
- **Relevance Scoring**: Computes cosine similarity scores between queries and documents.
- **Dimensionality Reduction**: Uses SVD and bidiagonalization with QR to reduce noise and improve search results.
- **Performance Analysis**: Compares execution times of standard, SVD, and bidiagonalization+QR methods.

## Technologies Used
- **Programming Language**: Python
- **Libraries**: NumPy, Matplotlib
- **Algorithms**: 
  - Singular Value Decomposition (SVD)
  - Bidiagonalization with QR factorization

## Project Structure
1. **Matrix Construction**:
   - `construire_matrice_td()`: Builds a term-document matrix.
   - `construire_vecteur_requete()`: Converts queries into vector form.

2. **Relevance Scoring**:
   - `calculer_scores_standard()`: Computes scores without SVD.
   - `calculer_scores_svd()`: Computes scores using SVD.

3. **Dimensionality Reduction**:
   - `decomposition_SVD()`: Direct SVD decomposition.
   - `approximation_rang_k()`: Rank-k approximation of a matrix.
   - `bidiagonalisation()` and `methode_qr_bidiagonale()`: Bidiagonalization and QR for SVD.

4. **Performance Analysis**:
   - `analyser_performances()`: Measures execution time for different matrix sizes.
   - `erreur_reconstruction()`: Computes reconstruction error.

5. **Examples and Tests**:
   - `exemple1()` and `exemple3()`: Demonstrate the system with predefined datasets.
   - `test_fichier_documents()`: Processes a text file to build a term-document matrix and tests retrieval.

## Code Examples
### Building a Term-Document Matrix
```python
def construire_matrice_td(termes, documents):
    D = np.zeros((len(termes), len(documents)), dtype=int)
    for j, doc in enumerate(documents):
        for i, terme in enumerate(termes):
            if terme in doc:
                D[i, j] = 1
    return D
```

### Computing Relevance Scores with SVD
```python
def calculer_scores_svd(U_k, S_k, Vt_k, q):
    q_proj = np.dot(U_k.T, q)
    norme_q_proj = np.linalg.norm(q_proj)
    scores = []
    for j in range(Vt_k.shape[1]):
        e_j = np.zeros(Vt_k.shape[1])
        e_j[j] = 1
        d_j_proj = np.dot(S_k, np.dot(Vt_k, e_j))
        score = np.dot(q_proj, d_j_proj) / (norme_q_proj * np.linalg.norm(d_j_proj))
        scores.append(score)
    return np.array(scores)
```

## How to Use
1. **Run Examples**:
   - Execute `exemple1()` or `exemple3()` to see the system in action with predefined datasets.
   - Use `test_fichier_documents("documents.txt")` to test with custom data.

2. **Performance Analysis**:
   - Run `analyser_performances()` to compare execution times across methods.

3. **Visualization**:
   - The script generates plots for reconstruction errors and performance metrics, saved as PNG files.

## Results
- **Standard Method**: Fast but less robust to noise.
- **SVD Direct**: Improved relevance with reduced dimensions.
- **Bidiagonalization + QR**: Alternative SVD method with trade-offs in speed and accuracy.

## Contributors
- Adem Medyouni
---

For detailed explanations or troubleshooting, refer to the project documentation or contact the contributor.
``` 
