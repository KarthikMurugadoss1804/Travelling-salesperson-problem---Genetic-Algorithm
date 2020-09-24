# -*- coding: utf-8 -*-
"""
Created on Sun Oct 20 02:13:50 2019

@author: karth
"""



"""
Basic TSP Example
file: Individual.py
"""

import random
import math

class Heuristic:
    def __init__(self, _size, _data):
        """
        Parameters and general variables
        """
        self.fitness    = 0
        self.genes      = []
        self.genSize    = _size
        self.data       = _data
        self.genes=self.insertion_heuristic1(self.data)
    
    def euclideanDistane(self,cityA, cityB):
    ##Euclidean distance
    #return math.sqrt( (cityA[0]-cityB[0])**2 + (cityA[1]-cityB[1])**2 )
    ##Rounding nearest integer
        return round( math.sqrt( (cityA[0]-cityB[0])**2 + (cityA[1]-cityB[1])**2 ) )


# Choose first city randomly, thereafter append nearest unrouted city to last city added to rpute
    def insertion_heuristic1(self,instance):
        cities = list(instance.keys())
        cIndex = random.randint(0, len(instance)-1)
    
        tCost = 0
    
        solution = [cities[cIndex]]
        
        del cities[cIndex]
    
        current_city = solution[0]
        while len(cities) > 0:
            bCity = cities[0]
            bCost = self.euclideanDistane(instance[current_city], instance[bCity])
            bIndex = 0
    #        print(bCity,bCost)
            for city_index in range(1, len(cities)):
                city = cities[city_index]
                cost = self.euclideanDistane(instance[current_city], instance[city])
    #            print(cities[city_index], "Cost: ",cost)
                if bCost > cost:
                    bCost = cost
                    bCity = city
                    bIndex = city_index
            tCost += bCost
            current_city = bCity
            solution.append(current_city)
            del cities[bIndex]
        tCost += self.euclideanDistane(instance[current_city], instance[solution[0]])
        return solution

        

   
        

    def setGene(self, genes):
        """
        Updating current choromosome
        """
        self.genes = []
        for gene_i in genes:
            self.genes.append(gene_i)

    def copy(self):
        """
        Creating a new individual
        """
        heu = Heuristic(self.genSize, self.data)
        for i in range(0, self.genSize):
            heu.genes[i] = self.genes[i]
        heu.fitness = self.getFitness()
        return heu

    def euclideanDistance(self, c1, c2):
        """
        Distance between two cities
        """
        d1 = self.data[c1]
        d2 = self.data[c2]
        return math.sqrt( (d1[0]-d2[0])**2 + (d1[1]-d2[1])**2 )

    def getFitness(self):
        return self.fitness

    def computeFitness(self):
        """
        Computing the cost or fitness of the individual
        """
        self.fitness    = self.euclideanDistance(self.genes[0], self.genes[len(self.genes)-1])
        for i in range(0, self.genSize-1):
            self.fitness += self.euclideanDistance(self.genes[i], self.genes[i+1])
            
