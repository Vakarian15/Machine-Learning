The program can be run in the terminal as such:

python3 ./em.py [file] [alg]

Where [file] is replaced with the path to a csv data file.
And [n_clusters] is replaced with an integer stating the number 
of clusters to be made. If the given number is 0 then the program 
will try to determine the optimal number of clusters on its own


An example of the command with values filled in:

python3 ./em.py data.csv 3

An example if the file name contains spaces:

python3 ./em.py "another data file.csv" 3

The following libraries will need to be installed:
numpy
argparse
scipy
time