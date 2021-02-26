The program can be run in the terminal as such:

python3 ./truck_driver.py [capacity] [length] [penalty] [clock_ticks]

Where [capacity] is replaced with a positive integer that represents the maximum number of packages to be loaded to the truck
[length] is replaced with a positive that defines the lengrh of the road
[penalty] is replaced with a negative integer that represents the cost of starting the truck without packages
and [clock_ticks]is replaced with a positive integer that limitis the number of clock ticks to train the package
    

An example of the command with values filled in:

python3 ./truck_driver.py 20 30 -250 1000

The following libraries will need to be installed:
numpy
argparse
