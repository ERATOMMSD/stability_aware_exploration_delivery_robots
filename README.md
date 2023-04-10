# Stability-aware Exploration of Design Space of Autonomous Robots for Goods Delivery
Autonomous robots have recently been employed for goods delivery, with the goal of reducing traffic congestion, pollution, and operational costs. The design of such a delivery service requires to select the number of robots, their operating hours, and speed. Requirements from different stakeholders must be considered: customer satisfaction, cost, and safety. To assist with said design, our industry partner Panasonic is employing a search-based approach that tries to find service configurations that optimise the three requirements, on average, across different possible sets of customer requests. The obtained Pareto fronts of solutions show the trade-off existing among the different requirements. Such Pareto fronts, albeit very useful, do not always facilitate an informed decision for the stakeholders, for they provide too many solutions (some of them very similar to each other). To tackle this issue, in this paper we propose two approaches to prune and simplify Pareto fronts. Our approaches consider the standard deviation of objective values across the different sets of customer requests; the intuition is that, if two solutions (expressed in terms of average objective values) overlap based on their standard deviations, they can be considered similar. Based on this intuition, the two pruning approaches group similar solutions and select only one representative for each partition. We assessed these pruning methods on the Pareto fronts obtained with the search-based approach employed by Panasonic. We found that they can significantly reduce the size of the Pareto fronts without reducing too much their quality (measured in terms of Hypervolume).

## People
* Mauricio Byrd Victorica https://www.kth.se/profile/mbv?l=en
* Paolo Arcaini http://group-mmm.org/~arcaini/
* Fuyuki Ishikawa http://research.nii.ac.jp/~f-ishikawa/en/
* Hirokazu Kawamoto
* Kaoru Sawai
* Eiichi Muramoto

## Paper
M. Byrd Victorica, P. Arcaini, F. Ishikawa, H. Kawamoto, K. Sawai, E. Muramoto. Stability-aware Exploration of Design Space of Autonomous Robots for Goods Delivery. In 27th International Conference on Engineering of Complex Computer Systems (ICECCS 2023).
