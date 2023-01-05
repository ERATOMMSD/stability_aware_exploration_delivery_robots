data - placeholder directory with a sample Pareto front as a CSV file.

helper.py: contains the implementations of STDV_BASED and STDV_DIST_BASED and necessary functions.

main.py: code to prune a given Pareto front. The data directory containing the front as a CSV file, the directory to output results, the K parameter for the K-overlap notion, and the epsilon parameter to be used for the STV_DIST_BASED method are to be passed to the script. A sample run with the default values is as follows:

From the root of this directory run:
python main.py --data_dir data/SampleFront.csv --K 0.3 --eps 0.0075
