import numpy as np
from sklearn.cluster import DBSCAN
from jmetal.core.solution import Solution
from jmetal.util.solution import get_non_dominated_solutions
import copy

def get_non_dom_from_numpy(solutions):
    """
    Computes the Pareto front (set of non dominated solutions) from a given set of solutions with objective values
    that are to be minimised.

    solutions	    np.array, solutions from which the non-dominated set is to be computed.

	Return 			np.array, non-dominated solutions or Pareto front.
    """
    real_solutions = []
    for row in solutions:
        solution = Solution(2, 3, 0)
        solution.objectives = list(row)
        real_solutions.append(solution)
    non_dominated_solutions = get_non_dominated_solutions(real_solutions)
    return np.array([np.array(sol.objectives) for sol in non_dominated_solutions])

def get_solutions_as_numpy_array(exp_df):
    """
    Formats a Pareto front given as a dataframe as a numpy array, most importantly objective values are modified
    to reflect the minimisation problem addressed (i.e. objectives that should be maximised are negated).

    exp_df	        Pandas dataframe, contains a set of original solutions.

	Return 			np.array, solutions as np.array and with the correct signs for each objective.
    """
    return np.array([exp_df['delivery_rate'] * -1.0, exp_df['utilization_rate'] * -1.0, exp_df['num_risks']]).transpose()



def metric_maker(varmethod=2, q=0.5):
    """
    Returns a distance function that considers K-overlapping when computing the
    regular distance between two points.

    varmethod	    int, indicates whether to consider std. dev over each objective (1),
                    all at once (2), or just compute the usual distance (0).
	q 		        float, number of standard deviations to consider for computing overlaps (K parameter).

	Return 			callable function, distance function which considers K-overlapping.
    """
    def dist_fun(x, y, method = varmethod, q = q):
        """
        Computes distances taking into account the variance used for clustering
        Pareto front solutions previously.

        x, y            np.array, two points for which distance is to be computed
        varmethod	    int, indicates whether to consider std. dev over each objective (1),
                        all at once (2), or just compute the usual distance (0).
    	q 		        float, number of standard deviations to consider for computing overlaps (K parameter).

        Return          float, distance between two points, according to the criteria to be considered.
        """
        #first option: K-overlapping over any objective
        if method==1:
            test=any([abs(x[z] - y[z])<\
                  q*abs(x[z+3]**0.5+y[z+3]**0.5)\
                  for z in range(3)])
        #second option: Regular K-overlapping
        elif method==2:
            test=all([abs(x[z] - y[z])<\
                  q*abs(x[z+3]**0.5+y[z+3]**0.5)\
                  for z in range(3)])
        elif method==0:
            test=False
        if test:
            return 0
        else:
            return abs(np.linalg.norm(x[:3]-y[:3]))
    return dist_fun


def get_labels(data, db):
    """
    Compute all labels from a clustering result using STDV_BASED.

    data	    np.array, the original data that was clustered by the STDV_BASED algorithm
	db 		    list, clusters found by STDV_BASED represented as lists of indexes from the data.

	Return 	    list, unique cluster labels determined by STDV_BASED.
    """
    labels=[]
    for i in range(len(data)):
        for l in range(len(db)):
            if i in db[l]:
                labels.append(l)
    return np.array(labels)

def STDV_BASED(data, metric):
    """
    Clustering algorithm STDV_BASED to cluster a Pareto front based only on the
    standard deviation of objective values.

    data	    np.array, the original data (Pareto front) to cluster.
	metric 		callable function, computes the distance between two points to perform clustering,
                for STDV_BASED, this function should be zero if the two points K-overlap for a
                predetermined K.

	Return 	    list, clusters found by STDV_BASED represented as lists of indexes from the data.
    """
    clusters=[]
    for i in range(len(data)):
        for j in range(len(data)):
            if i==j:
                continue
            test = metric(data[i], data[j])==0
            if test and [j,i] not in clusters:
                clusters.append([i,j])
    for i in range(len(data)):
        found=False
        for c in clusters:
            if i in c:
                found=True
                break
        if found:
            continue
        else:
            clusters.append([i])

    convergence=False
    while(not convergence):
        convergence=True
        for c in clusters:
            merge=False
            for k in clusters:
               if c==k:
                   continue
               if any([x in k for x in c]):
                   merge=True
                   convergence=False
                   gift=[n for n in set(c+k)]
                   clusters.remove(c)
                   clusters.remove(k)
                   clusters.append(gift)
                   break
            if merge:
                break
    overlaps = copy.deepcopy(clusters)
    for c in clusters:
        overlaps.remove(c)
        for k in overlaps:
            if k==c:
                clusters.remove(c)
        overlaps = copy.deepcopy(clusters)

    return clusters

def STDV_DIST_BASED(data, eps=0.5, min_samples=1, metric=metric_maker(varmethod=2, q=0.5)):
    """
    Clustering algorithm STDV_BASED_DIST to cluster a Pareto front based on the
    standard deviation of objective values and their regular distance using taking
    the DBSCAN algorithm as the foundation.

    data	        np.array, the original data (Pareto front) to cluster.
    eps             float, the epsilon parameter considered to cluster points by distance,
                    input to the DBSCAN algorithm.
    minsamples      int, another parameter for DBSCAN, indicating how many neighbours
                    are needed to consider a sample as a "core sample".
    metric 		    callable function, computes the distance between two points to perform clustering,
                    for STDV_BASED, this function should be zero if the two points K-overlap for a
                    predetermined K.

	Return 			callable function, distance function which considers K-overlapping.
    """
    return DBSCAN(eps=eps, min_samples=min_samples, metric=metric).fit(data)
