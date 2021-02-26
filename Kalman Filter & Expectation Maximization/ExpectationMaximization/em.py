import numpy as np
import argparse
from scipy.stats import norm, gaussian_kde
import time


#####################################
######### Class Definitions #########
#####################################

class Centroid(object):
    """ A centroid for clustering.
    Attributes:
        values (numpy array): values for the centroid, len(values) is dimensionality of the data
        variance (numpy array): variance for the centroids distribution, len(variance) is dimensionality of the data
        id (int): N-th cluster, helpful value to update the mean and variance of the centroid
    """
    def __init__(self, values, variance, id):
        self.values = values 
        self.variance = variance
        self.id = id

    def update_values_variance(self, data, prob_array):
        """
        Attributes:
            data (numpy array): n by d array. Where n is the number of observations and d dimensions
            prob_array (numpy array): Probability that a given observation belongs to a particular cluster. n by c array. n is the number of observations and c the number of clusters.
        """
        #Compute sum of probabilities to normalize result
        sum_ = np.sum(prob_array, axis=0)[self.id]
        
        #Calculate means with the given probabilities
        for dimension in range(data.shape[1]):
            self.values[dimension] = (data.T[dimension].dot(prob_array.T[self.id]))/sum_
        
        #Copy of data to substract the centroid values and calculate variance
        data_ = data.copy()
        
        #Calculate variance with the given probabilities
        for dimension in range(data_.shape[1]):
            data_.T[dimension] = (data_.T[dimension] - self.values[dimension])**2
            self.variance[dimension] = (data_.T[dimension].dot(prob_array.T[self.id]))/sum_
              
#####################################
######### Helper  Functions #########
#####################################

#-----------------------------------
def obtain_one_cluster(y):
    """ Approxiamte the possible number of clusters per one dimension
    Args:
        y (numpy array): n by 1 array
    Returns:
        max_ (int): the unmber of peaks at the normal distribution of array y
    """
    kernel = gaussian_kde(y)
    #Generate 50 points based on a normal kernel of the data
    y_k = kernel.logpdf(np.linspace(min(y), max(y), num=50))

    max_ = 0 
    #Count the number of peaks as an approximation of the number of clusters
    for i in range(1, len(y_k)-1):
        if y_k[i] > y_k[i+1] and y_k[i] > y_k[i-1]:
            max_ += 1
    return max_

#-----------------------------------
def obtain_k_cluster(y):
    """ Approxiamte the number of clusters in n dimensions
    Args:
        y (numpy array): n by d array. n observations in d dimensions
    Returns:
        max_ (tuple): a maximum of three possibilities of the number of cluster in the data
    """
    clusters = [obtain_one_cluster(y.T[i]) for i in range(y.shape[1])]
    max_ = max(clusters)
    if max_ == 0:
        return 1
    else:
        return max_

#-----------------------------------
def get_random_start(observations, n_clusters):
    """ Randomly create n_clusters from the dataset.
    Args:
        observations (numpy array): the provided data points 
        n_clusters (int): number of centroids to make
    Returns:
        (list): a list of Centroid objects
    """
    dimension = data.shape[1]
    centroids = []

    if n_clusters > 1:
        # Choose n_clusters number of observations to act as centroids
        c_observations = data[np.random.choice(len(data), n_clusters, replace=False)]

        # Set same variance for all centroids. It equals a contant value times the total variance of a given dimension
        std = np.array([data.T[i].std()*np.random.uniform(.70,.95) for i in range(dimension)])

        for i in range(n_clusters):
            # FIXME: think of other alternative
            centroids.append(Centroid(c_observations[i], std**2, i))
        return centroids

    else:
        centroids.append(Centroid(np.mean(data,axis=0), np.std(data,axis=0)**2, 0))
        return centroids

#-----------------------------------
def get_normal_distributions(centroid): 
    """ Return a list of normal distributions based on one Centroid object
        Each normal corresponds to one dimension of the data
        Dimensions of each Centroid are independent
    Args:
        centroids: a Centroid object 
    Returns:
        (list): a list of scipy objects that refer to a normal distribution
    """
    normals = []

    #Loop for each dimension
    for i in range(len(centroid.values)):
        normals.append(norm(loc=centroid.values[i], scale=centroid.variance[i]**0.5))

    return normals

#-----------------------------------
def get_probability_cluster(observation, normals):
    """ The probability to belong to a cluster, that equals the product of the probabilities to belong to each dimension 
    Args:
        observation (numpy array): a observation of the data, an array of shape (d,) 
            where d is the number of dimensions of the data
        normals (list): a list of scipy objects that refer to a normal distribution for each dimension
    Returns:
        prod_probability (float): the probability to belong to a particular cluster 
    """
    prod_log_probability = 1 

    #Loop for each dimension
    for i in range(len(observation)):
        prod_log_probability *= normals[i].pdf(observation[i]) 

    return prod_log_probability 

#-----------------------------------
def get_normalized_probability_cluster(prob_array):
    """ Normalize the probabilities to belong to one cluster 
    Args:
        prod_log_probability (numpy array): non normalized probability array
    Returns:
        probability (numpy array): normalized probability array where the (i,j) elements corresponds
            to the probability to instance i to belong to cluster j
    """
    if prob_array.shape[1] == 1:
        return prob_array
    else:    
        sums = 1 / np.sum(prob_array, axis=1) 

        #Loop for each instance
        for i in range(len(sums)):
            prob_array[i] =  sums[i]*prob_array[i]

        return prob_array # Do not consider the sign, since it depends on the number of dimensions

#-----------------------------------
def classification(prob_array):
    """ Classify the observations 
    Args:
        prob_array (numpy array): n by k array, where element (i,j) represents the probability of observation i to belong to j cluster 
    Returns:
        classificaction (numpy array): array of n elements, the ith number corresponds to the cluster of the ith observation in the data
    """
    if prob_array.shape[1] == 1:
        return np.zeros(len(prob_array), dtype=int)

    prob_array_copy = prob_array.copy() 

    # For each observation
    for i in range(1, prob_array_copy.shape[1]):
        prob_array_copy.T[i] = prob_array_copy.T[i] + prob_array_copy.T[i-1]
    
    # Create an array of random variables to select the cluster of each observation 
    dice = np.random.uniform(size=prob_array_copy.shape[0])

    # Classify each observation based on the result of the random values
    cluster = []
    for i in range(len(dice)):
        j = np.searchsorted(prob_array_copy[i], dice[i])
        cluster.append(j)
    cluster = np.array(cluster)

    return cluster   

#-----------------------------------
def write_results(data, classified):
    """ Write results
    Args:
        data (numpy array): n by d array
        classified (numpy array): (n,) array where the nth value refers to the cluster that corresponds to the data
    """
    out = np.column_stack((data, classified))

    #Create string for creating the csv file
    title = ""
    for i in range(data.shape[1]):
        title += 'X'+str(i+1)+','
    title += 'Cluster'

    #Save csv file
    np.savetxt("out.csv", out, delimiter=',', header=title, comments="")

#-----------------------------------
def get_log_BIC(prob_array, n_clusters, data_shape_1):
    """ Calculate log likelihood and BIC 
    Args:
        classified (numpy array): array of lenght n, where the nth element corresponds to the k cluster that includes the nth point
        prob_array (numpy array): n by k array, where element (i,j) represents the probability of observation i to belong to j cluster 
    Returns:
        ll (int): Log likelihood of the model
        BIC (int): Bayesian information criterion
    """
    # Calculate log likelihood
    ll = 0
    for i in range(len(prob_array)):
        ll += np.log(max(prob_array[i]))

    # Calculate BIC
    BIC = np.log(len(prob_array))*(2*n_clusters*data_shape_1) - 2*ll

    return ll, BIC

#-----------------------------------
def parse_command_line():
    """ Parse the command line arguments.
    Returns:
        (Namespace Object): object with attributes 'file' and 'n_clusters'
    """
    parser = argparse.ArgumentParser()
    parser.add_argument('file', type=str, help='Path to file containing data to be clustered')
    parser.add_argument('n_clusters', type=int, help='The number of clusters in the data')
    
    return parser.parse_args()

######################################
## Estimation Maximization Function ##
######################################

def estimation_maximization(observations, centroids):
    """ Perform estimation maximixation.
    Args:
        observations (np array): an array of the data
        centroids (list): a list of Centroid object
    Returns:
        (list): the list of Centroid objects which now have updated values lists
    """ 
    #Generate a numpy array corresponding to the probabilities to belong to each cluster
    prob_array = np.zeros((len(observations), len(centroids)))

    for cluster, centroid in enumerate(centroids):
        # Create the normal distribution for each dimension
        normals = get_normal_distributions(centroid)

        for i, obs in enumerate(observations):
            # Determine the probability that the observation could come from the centroids distribution
            prob_array[(i, cluster)] = get_probability_cluster(obs, normals)

    prob_array = get_normalized_probability_cluster(prob_array)

    #Update the means of the centroids if the number of centroids is bigger than 1
    if len(centroids) > 1:
        for centroid in centroids:
            centroid.update_values_variance(observations, prob_array)

    return centroids, prob_array

#####################################
######## Main Program Driver ########
#####################################

if __name__ == "__main__":

    #Create time variable
    start_time = time.time()
    
    #Parse arguments and create data
    args = parse_command_line()
    data = np.loadtxt(args.file, delimiter=",") 
    
    #----- Decide the best number of clusters
    if args.n_clusters == 0:

        # Have a initial aproximation of the number of clusters
        n_clusters = obtain_k_cluster(data)
        
        # Make random centroids to start
        centroids_0 = get_random_start(data, n_clusters)
        centroids_1 = get_random_start(data, n_clusters + 1)

        # First series of loop: 5 steps or less than 5 seconds
        count = 0
        while time.time() - start_time < 5 and count < 6:
            centroids_0, prob_array_0 = estimation_maximization(data, centroids_0)
            centroids_1, prob_array_1 = estimation_maximization(data, centroids_1)
            count += 1

        # Calculate Log Likelihood
        BIC_0 = get_log_BIC(prob_array_0, n_clusters, data.shape[1])[1]
        BIC_1 = get_log_BIC(prob_array_1, n_clusters+1, data.shape[1])[1]
        
        # Change variable names to be consistent with later calculations
        if BIC_1 < BIC_0:
            centroids = centroids_1
            n_clusters += 1
        else:
            centroids = centroids_0

        # Loop while there is still time left
        while time.time() - start_time < 10:
            centroids, prob_array = estimation_maximization(data, centroids)
    
    #----- Run the algorithm with the selected number of clusters
    else:
        n_clusters = args.n_clusters

        # Make random centroids to start
        centroids = get_random_start(data, n_clusters)

        # If the number of clusters is 1
        if n_clusters == 1:
            centroids, prob_array = estimation_maximization(data, centroids)
        
        else:
            # Loop while there is still time left
            while time.time() - start_time < 10:
                centroids, prob_array = estimation_maximization(data, centroids)
    #-----
    # Classify the observations
    classified = classification(prob_array)

    # Calculate Log Likelihood
    ll, BIC = get_log_BIC(prob_array, n_clusters, data.shape[1])
    #-----

    # Print results
    print()
    with np.printoptions(precision=3):
        for i in range(n_clusters):
            print("Cluster", i+1)
            print('\t', "Mean:", centroids[i].values)
            print('\t', "Variance:", centroids[i].variance)
    print('\n',"Log-likelihood: ", '%.3f' % ll, sep="")
    print('\n',"BIC: ", '%.3f' % BIC, sep="")     
    print()
    #Create out data
    write_results(data, classified + 1)