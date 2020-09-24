# Travelling salesperson problem - Genetic Algorithm
 In this project we intend to solve the classic traveling salesperson problem using genetic algorthim.
 
 ## Refer Project Description for detailed information about the project
 
   
## Files used

**Folder: Source Code Files**

*********************************

**1) TSP_R00183157.py**

This file contains code for configuration 1-6 and it takes five arguments

a) data file (.tsp file)

b) population size

c) mutation

d) iterations

e) configuration number between (1-6)

example:

python TSP_R00183157.py inst-4.tsp 100 0.1 500 1

**2) TSP_R00183157_elite.py**

This file contains code implementation for elite survival for the configuration 1-6 and it takes 6 arguments

a) data file (.tsp file)

b) population size

c) mutation

d) iterations

e) configuration number between (1-6)

f) elite percentage

example for execution:

python TSP_R00183157_elite.py inst-4.tsp 100 0.1 500 2 0.2

**3) TSP_R00183157_heuristic.py**

This file contains code implementatio for heuristically selected population( config 7-8) and it takes four arguments

a) data file (.tsp file)

b) population size

c) mutation

d) iterations

example for execution:

python TSP_R00183157_heuristic.py inst-4.tsp 100 0.1 500

**4)Heuristic.py**

This file contains the code with class heuristic which is imported in the main TSP to implement heuristic population initialization and is required for implementing it.

**5) Results_configuration.xlsx**

This spreadsheet contains all results combined together based on which the algorithms were analysed.

**************************************************************

**Folder: Output files**

************************************************************

This folder contains all the output files taken from all runs which was taken and consolidated into the spreadsheet

*************************************************************

**Folder: Data Files**

************************************************************

This folder contains files inst-4.tsp, inst-6.tsp and inst-16.tsp which are the input data files for this project
