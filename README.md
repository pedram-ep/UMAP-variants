Basic UMAP (Uniform Manifold Approximation and Projection) has limitations for specific tasks like projecting new data in the low-dimensional space. Parametric UMAP and Approximate UMAP are two variants of UMAP which want to solve this problem. Parametric UMAP uses a neural network as the mapping function, and Approximate UMAP uses a knn model to approximate new data points position in the low-dimensional space.
  

## Parametric UMAP

This method constructs the fuzzy simplicial complex (probabilistic graph) same as original UMAP. But instead of applying a graph optimization to find the low-dimensional graph, a neural network is trained as the mapping function $f: \mathbb{R}^{D} \rightarrow \mathbb{R}^{d}$. The architecture of this network, often referred to as the encoder, can vary significantly depending on the nature of the data being analyzed. For instance, simple tabular data might be effectively handled by a multi-layer perceptron (MLP), while image data could benefit from a convolutional neural network (CNN), and sequential data might require a recurrent neural network (RNN). The output layer of this network is designed to have a dimensionality that matches the desired dimensionality of the embedding space (e.g., two or three for visualization purposes).

This neural network uses this loss function:

$$\sum\limits_{i\neq j} p_{ij} \log \left(\frac{p_{ij}}{q_{ij}}\right) + (1 - p_{ij})\log\left(\frac{1-p_{ij}}{1-q_{ij}}\right)$$

- $p_{ij}$: Probability that points $i$ and $j$ are neighbors in the original data.
- $q_{ij}$: Probability that they’re neighbors in the embedding.

In the loss function, $p_{ij} \log \left(\frac{p_{ij}}{q_{ij}}\right)$ is the attraction term and pulls the neighbors closer in the embedding. And $(1 - p_{ij})\log\left(\frac{1-p_{ij}}{1-q_{ij}}\right)$  is the repulsion term which pushes non-neighbors apart in the embedding.

Once the neural network has been successfully trained, the process of embedding new data becomes remarkably efficient. To embed a new data point, it is simply passed through the trained neural network, and the output of the network represents its coordinates in the learned low-dimensional embedding space. This process is significantly faster than having to re-run the entire standard UMAP algorithm on the new data, making Parametric UMAP highly advantageous in dynamic or large-scale data scenarios.

## Approximate UMAP
Approximate UMAP performs standard UMAP training on the initial dataset, then trains an auxiliary kNN model for fast queries. The novel embedding step for each new point $x$ is as $u$ as:

$$u = \sum\limits_{i=1}^{k}\frac{\frac{1}{d_{i}}}{\sum\limits_{j=1}^{k} \frac{1}{d_{j}}} u_{i}$$

where $k$ is the hyperparameter of number of neighbors, $u_{1}, \cdots, u_{k}$ are projections of $x_{1}, \cdots, x_{k}$ from the base dataset, and $d_{i}(x) = \text{distance}(x, x_{i})$ is a distance function.

The main strength of Approximate UMAP is speed: projecting a new point takes only a kNN lookup and averaging, which is orders of magnitude faster than UMAP’s iterative embedding. Since aUMAP’s solution _exactly_ minimizes distance to neighbors’ embeddings, its output closely approximates the UMAP projection (experiments show only tiny deviations). The key limitation is approximation error: new points may not be optimally placed if the average of neighbors does not satisfy UMAP’s cost exactly, especially if the data manifold is highly curved or anisotropic. Also, Approximate UMAP assumes that the learned UMAP embedding is good; if the initial embedding had distortions, Approximate UMAP simply inherits them. Unlike parametric UMAP, Approximate UMAP does not generalize beyond averaging: it cannot adapt to entirely new regions of the manifold or “learn” a mapping. Finally, Approximate UMAP requires storing all training embeddings and a kNN data structure, which can be memory-intensive for very large datasets.

  
# Dataset
This project uses the MNIST dataset (Modified National Institute of Standards and Technology database), a classic benchmark dataset in machine learning. The dataset contains:
- 70,000 handwritten digit images (0-9)
- 28×28 pixel grayscale images (flattened to 784-dimensional vectors)
- Preprocessed with:
    - Pixel values normalized to [0, 1] range (`X = X.astype(np.float32) / 255.0`)
    - Labels included for potential evaluation (though not used in the core UMAP implementations)

The dataset is automatically downloaded via scikit-learn's `fetch_openml` using:
```py
from sklearn.datasets import fetch_openml
X, y = fetch_openml('mnist_784', version=1, return_X_y=True, as_frame=False)
```
Source: [OpenML MNIST_784](https://www.openml.org/d/554)
# Requirements

- Python 3.6+
- Required packages:
```
numpy>=1.20
scikit-learn>=1.0
umap-learn>=0.5
tensorflow>=2.6
```
- For ApproxUMAP (install from GitHub):
```
git+https://github.com/learnedsystems/approx_umap.git
```

# Installation
```bash
git clone https://github.com/pedram-ep/UMAP-variants
cd UMAP-variants
pip install -r requirements.txt
```
  
  

# Refrences

1. Wassenaar P, Guetschel P, Tangermann M. Approximate UMAP allows for high-rate online visualization of high-dimensional data streams. 2024. | [arxiv link](https://arxiv.org/abs/2404.04001)
2. McInnes L, Healy J, Melville J. UMAP: Uniform manifold approximation and projection for dimension re-duction. 2018. | [arxiv link](https://arxiv.org/abs/1802.03426)
3. Sainburg T, McInnes L, Gentner TQ. Parametric UMAP embeddings for representation and semisuper-vised learning. Neural Computation. 2021. [arxiv link](https://arxiv.org/abs/2009.12981)
# Related Papers
1. Ghojogh B, Ghodsi A, Karray F, Crowley M. Uniform Manifold Approximation and Projection (UMAP) and its Variants: Tutorial and Survey. 2021. | [arxiv link](https://arxiv.org/abs/2109.02508)
