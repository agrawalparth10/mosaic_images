import numpy as np

class K_Mediods():

    def __init__(self, X, n_clusters):
        '''
        K Mediods class 

        Args:
            X : numpy array [B x N] 
                where 
                    B is the number of images
                    N = H x W x C
                Linearized images
            n_clusters : int
                Number of clusters
        '''

        self.X = X
        self.n_clusters = n_clusters
        n_samples, _ = X.shape

        # Random create mediods
        self.mediods = np.random.choice(n_samples, n_clusters, replace=False)

        # Assign the clusters to each image
        self.assign_clusters()

        # Find the optimal solution
        self.converge()
        
    
    def create_mediods(self):
        '''
        Create a numpy array of mediods

        Returns:
            mediods_arr : numpy array (num_clusters x N)
                numpy array of mediods
        '''

        # Find the array from X
        mediods_arr = self.X[self.mediods]
        return mediods_arr
        
    def assign_clusters(self):
        '''
        Assign clusters to all the images
        '''

        # Create the mediods array
        mediods_arr = self.create_mediods()

        # Find the distances between mediods and images
        dists = self.l2_distances(mediods_arr, self.X)

        # Assign labels to all the images
        self.labels = self.predict_labels(dists)
        return

    def l2_distances(self, X, Y):
        '''
        Calculate the l2 distances between the arrays

        Args:
            X : numpy array
            Y : numpy array
        
        Returns:
            dists : numpy array 
                Distances between each instance of X and Y
        '''

        num_Y = Y.shape[0]
        num_X = X.shape[0]
        dists = np.zeros((num_Y, num_X))
        dists = np.sqrt(abs(np.sum(np.square(Y)[:,np.newaxis,:], axis=2) - 2 * Y.dot(X.T) + np.sum(np.square(X), axis=1)))

        return dists
    
    def predict_labels(self, dists, k=1):
        '''
        Assign labels to closest mediod

        Args:
            dists : numpy array
                Distances between each instance of X and Y
            k : int
                number of closest instances

        Returns:
            y_pred : numpy array 
                Indices of the closest mediods
        '''

        num_Y = dists.shape[0]
        y_pred = np.zeros(num_Y)
        for i in np.arange(num_Y):
       
            closest_y = []
            dists_val = dists[i]
            sort_index = np.argsort(dists_val)
            sort_index = sort_index[:k]
            closest_y = self.mediods[sort_index]
            most_common = np.bincount(closest_y).argmax()
            y_pred[i] = np.where(self.mediods==most_common)[0][0]

        return y_pred
        
    def get_medoid(self, cluster) :
        '''
        Find the mediod of the cluster

        Args:
            cluster: numpy array
        
        Returns:
            Index of mediod in the cluster
        '''

        # Calculate distances
        dists = self.l2_distances(cluster, cluster)
        sum_dists = np.sum(dists,axis = 1,keepdims = True)

        # Index of the minimum distance
        index = np.argmin(sum_dists)
        return index

    def get_cluster(self, cluster_num):
        '''
        Returns the specific cluster

        Args:
            cluster_num : int
                cluster number

        Returns:
            numpy array
        '''

        # Incidices of images in the cluster
        indicies = np.squeeze(np.argwhere(self.labels == cluster_num),axis=1)

        # Index the images
        cluster = self.X[indicies]
        return cluster


    def converge(self):
        '''
        Apply the k mediods algorithm
        '''
        iter = 0
        while(True):
            
            # Store old labels to check for covergence
            old = self.labels

            # Loop over all the clusters
            for i in range(self.n_clusters):

                # Find the cluster
                cluster = self.get_cluster(i)

                # Find the new mediod of the cluster
                index = self.get_medoid(cluster)
                cluster_indicies = np.squeeze(np.argwhere(self.labels == i),axis=1)
                self.mediods[i] = cluster_indicies[index]
            
            # Assign clusters to images with new mediods
            self.assign_clusters()
            
            iter += 1

            # Break if converged
            if np.array_equal(old, self.labels) or iter > 10:
                break
        return
    
    def pred(self, X):
        '''
        Find the cluster of X and return the index of 
        a random image in the cluster
        
        Args:
            X : numpy array
            
        Returns:
            index : int
        '''

        # Create the mediods array
        mediods_arr = self.create_mediods()

        # Calculate distance of X from mediods
        dists = self.l2_distances(mediods_arr, X)

        # Assign labels
        labels = self.predict_labels(dists)[0]

        # Find the images in the assigned cluster
        cluster_indicies = np.squeeze(np.argwhere(self.labels == labels),axis=1)

        # Return the index of a random image in the cluster
        index = np.random.choice(cluster_indicies,1)
        return index
