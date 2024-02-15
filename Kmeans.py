import numpy as np 
import matplotlib.pyplot as plt

def euclidian_distance(x1, x2):
    return np.sum(np.sqrt((x1-x2)**2))

class Kmeans:
    '''
    A K means classifier class.

    Parameters
    ----------
    K: int
        The number clusters to generate when predicting labels.
    max_iters: int
        Maximum number of cycles before algorithm stops. (May stop earlier if stabilizes.) 
    plot_steps: boolean
        When True the clsssifier will plot each step of the prediction cycles.

    Methods
    -------
    predict(X)
        Predict target values on a set of features X.

    '''
    def __init__(self, K=5, max_iters=100, plot_steps=False):
        self.K = K
        self.max_iters = max_iters
        self.random_generator = np.random.default_rng()
        self.plot_steps = plot_steps

        # List of sample indices for each cluster.
        self.clusters = [[] for _ in range(self.K)]

        # The centers (mean vector) for each cluster.
        self.centroids = []


    def predict(self, X):
        '''
        Predict label values given array of features X.

        Parameters
        ----------
        X: np.array of shape (n_samples, n_features)

        Returns
        -------
        Predictions: 1-d np.array of predicted labels.

        ''' 
        self.X = X
        self.n_samples, self.n_features = X.shape

        # Initialise the centroids.
        random_sample_idxs = self.random_generator.choice(self.n_samples, self.K, replace=False)
        self.centroids = [X[idx] for idx in random_sample_idxs]

        # Optimize the clusters.
        for _ in range(self.max_iters):
            # Assign the samples to their closest centroid (create the clusters)
            self.clusters = self._create_clusters(self.centroids)

            if self.plot_steps:
                # Plot the clusters.
                self.plot()

            # Calculate new centroids from the clusters.
            centroids_old = self.centroids
            self.centroids = self._get_centroids(self.clusters)

            # End if centroids have converged.
            if self._is_converged(centroids_old, self.centroids):
                break

            if self.plot_steps:
                # Plot new centroids.
                self.plot()

        # Classify the samples as the index of their cluster.
        return self._get_cluster_labels(self.clusters)
    
    def _create_clusters(self, centroids):
        # Assign each samples to its closest centroid.
        clusters = [[] for _ in range(self.K)]
        for idx, sample in enumerate(self.X):
            centroid_idx = self._closest_centroid(sample, centroids)
            clusters[centroid_idx].append(idx)
        return clusters

    def _closest_centroid(self, sample, centroids):
        # Calculate distance of sample to each centroid and return index of the minimum centroid.
        distances = [euclidian_distance(sample, centroid) for centroid in centroids]
        closest_centroid_idx = np.argmin(distances)
        return closest_centroid_idx

    def _get_centroids(self, clusters):
        # Assign the mean value of each cluster to its centroid.
        centroids = np.zeros((self.K, self.n_features))
        for cluster_idx, cluster in enumerate(clusters):
            cluster_mean = np.mean(self.X[cluster], axis=0)
            centroids[cluster_idx] = cluster_mean
        return centroids

    def _is_converged(self, centroids_old, centroids):
        # If no distance between old and current centroids, then converged.
        distances = [euclidian_distance(centroids_old[i], centroids[i]) for i in range(self.K)]
        return sum(distances) == 0

    def _get_cluster_labels(self, clusters):
        # Each sample is given the label of the cluster it was assigned to.
        labels = np.empty(self.n_samples)
        for cluster_idx, cluster in enumerate(clusters):
            for sample_idx in cluster:
                labels[sample_idx] = cluster_idx
        return labels
    
    def plot(self):
        fig, ax = plt.subplots(figsize=(12, 8))

        for cluster in self.clusters:
            points = self.X[cluster].T
            ax.scatter(*points)

        for point in self.centroids:
            ax.scatter(*point, marker='x', color='black', linewidth=3)
        
        plt.show()