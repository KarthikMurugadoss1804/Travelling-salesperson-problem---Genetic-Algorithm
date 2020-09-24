

"""
Author:
file:
Rename this file to TSP_x.py where x is your student number 
"""

import random
import datetime
from Individual import *
import sys


myStudentNum = 183157 # Replace 12345 with your student number
random.seed(myStudentNum)

class BasicTSP:
    def __init__(self, _fName, _popSize, _mutationRate, _maxIterations, _config):
        """
        Parameters and general variables
        """

        self.population     = []
        self.matingPool     = []
        self.best           = None
        self.popSize        = int(_popSize)
        self.genSize        = None
        self.mutationRate   = float(_mutationRate)
        self.maxIterations  = int(_maxIterations)
        self.iteration      = 0
        self.fName          = _fName
        self.data           = {}
        self.counter        = 0
        self.config         = int(_config)

        self.readInstance()
        self.initPopulation()


    def readInstance(self):
        """
        Reading an instance from fName
        """
        file = open(self.fName, 'r')
        self.genSize = int(file.readline())
        self.data = {}
        for line in file:
            (id, x, y) = line.split()
            self.data[int(id)] = (int(x), int(y))
        file.close()
#        print(self.data)

    def initPopulation(self):
        """
        Creating random individuals in the population
        """
        
        for i in range(0, self.popSize):
            individual = Individual(self.genSize, self.data)
            individual.computeFitness()
            self.population.append(individual)
#            print("Parent{}:{}\nFitness:{}\n".format(i+1,individual.genes[:],individual.fitness))
            

        self.best = self.population[0].copy()
        for ind_i in self.population:
            if self.best.getFitness() > ind_i.getFitness():
                self.best = ind_i.copy()
        print ("Best initial sol: ",self.best.getFitness())
#        print ("Best Initial Route:", self.best.genes[:])

    def updateBest(self, candidate):
        
        if self.best == None or candidate.getFitness() < self.best.getFitness():
            self.best = candidate.copy()            
            print("Counter resetting at iteration:",self.counter)
            self.counter=0
            
            print ("iteration: ",self.iteration, "best: ",self.best.getFitness())
        
            
            
            

    def randomSelection(self):
        """
        Random (uniform) selection of two individuals
        """
        indA = self.matingPool[ random.randint(0, self.popSize-1) ]
        indB = self.matingPool[ random.randint(0, self.popSize-1) ]
        return [indA, indB]

    def stochasticUniversalSampling(self):
        """
        Your stochastic universal sampling Selection Implementation
        
        """
        #Calculation of New Fitness and Sum of Fitness
        new_fitness=[] # 
        sum_of_fitness=0
        unit=1/self.popSize
#        print("Unit:",unit)
        for i in range(self.popSize):
            
            calc_fitness=1/(self.matingPool[i].fitness)
            sum_of_fitness+=calc_fitness
            new_fitness.append(calc_fitness)
        #Updating new Fitness with probability
        list_of_fitness=[]
        prob_fitness=[new_fitness[x]/sum_of_fitness for x in range(self.popSize)]
        list_of_fitness.append(prob_fitness[0])
        for i in range(1,self.popSize):
            list_of_fitness.append(prob_fitness[i]+list_of_fitness[i-1])
        rand_starting_point=random.uniform(0,1/self.popSize)
        biased_mating_pool=[]
        temp=rand_starting_point
        for x in range(self.popSize):
            temp+=unit
            if temp>1:temp-=1            
            for y in range(self.popSize):
                if(temp<=list_of_fitness[y]):
                    biased_mating_pool.append(self.matingPool[y])
                    break
        
        i1 = biased_mating_pool[ random.randint(0, self.popSize-1) ]
        i2 = biased_mating_pool[ random.randint(0, self.popSize-1) ]
        
        return i1,i2
       
    def uniformCrossover(self, indA, indB):
        """
        Your Uniform Crossover Implementation
        """
        exchange_genes=[random.randint(0,1) for i in range(self.genSize)]
        #Creating inital child chromosome
        child1=[indA.genes[x] if exchange_genes[x]==1 else None for x in range(self.genSize)]
#        child2=[indB.genes[x] if exchange_genes[x]==1 else None for x in range(self.genSize)]
        #Filling the remaining genes in the chromosome
        def douniformcrossover(child, parent):
            temp=parent.genes[:]
            for x in range(self.genSize):
                if child[x]==None:
                    for y in range(len(temp)):
                        if(temp[y] not in child and child[x]==None):
                            child[x]=temp[y]
                            del temp[y]
                            break        
            return child
        return douniformcrossover(child1,indB)#, douniformcrossover(child2, indA)

       

    def pmxCrossover(self, indA, indB):
        """
        Your PMX Crossover Implementation
        """
        #Creating Random Strip
        indexA=random.randint(0,self.genSize)
        indexB= random.randint(0,self.genSize)
        #Flipping the strip between Parent1 and Parent2
        child1=[indB.genes[i] if (i >= min(indexA, indexB) and i <= max(indexA, indexB)) else None for i in range(self.genSize)]
        child2=[indA.genes[i] if (i >= min(indexA, indexB) and i <= max(indexA, indexB)) else None for i in range(self.genSize)]
        #Creating mapping dictionary
        l1=[i for i in child1 if i]
        l2=[i for i in child2 if i]            
        d1=dict(zip(l1,l2))   
        d2=dict(zip(l2,l1))
      
      	#populating values in the child

        def domapping(child,parent,map_dict):
            for x in range(self.genSize):
                if child[x] is None and parent.genes[x] not in child:
                    child[x]=parent.genes[x]
                elif child[x] is None:
                    iter_value=parent.genes[x]
                    while child[x] is None:
                        iter_value=map_dict.get(iter_value)
                        if iter_value not in child:
                            child[x]=iter_value
            return child
        
             
        return domapping(child1,indA,d1) #, domapping(child2,indB,d2)
        
        
        
    def reciprocalExchangeMutation(self, ind):
        """
        Your Reciprocal Exchange Mutation implementation
        """
        pass

    def inversionMutation(self, ind):
        """
        Your Inversion Mutation implementation
        """
        if random.random() > self.mutationRate:
            return ind
        indexA=random.randint(0,self.genSize-1)
        indexB=random.randint(0,self.genSize-1)
        ma=max(indexA,indexB)
        
        child=[]
        
        for x in range(self.genSize):
            if x >=min(indexA,indexB) and x<=max(indexA,indexB):
                child.append(ind.genes[ma])
                ma-=1
                
                
            else:
                child.append(ind.genes[x])
        ind.setGene(child)
        ind.computeFitness()
        self.updateBest(ind)
        return ind
        
        

    def crossover(self, indA, indB):
        """
        Executes a 1 order crossover and returns a new individual
        """
        child = []
        tmp = {}

        indexA = random.randint(0, self.genSize-1)
        print(indexA)
        indexB = random.randint(0, self.genSize-1)
        print(indexB)
        for i in range(0, self.genSize):
            if i >= min(indexA, indexB) and i <= max(indexA, indexB):
                tmp[indA.genes[i]] = False
            else:
                tmp[indA.genes[i]] = True
        aux = []
        for i in range(0, self.genSize):
            if not tmp[indB.genes[i]]:
                child.append(indB.genes[i])
            else:
                aux.append(indB.genes[i])
        child += aux
        return child

    def mutation(self, ind):
        """
        Mutate an individual by swaping two cities with certain probability (i.e., mutation rate)
        """
        if random.random() > self.mutationRate:
            return ind
        indexA = random.randint(0, self.genSize-1)
        indexB = random.randint(0, self.genSize-1)

        tmp = ind.genes[indexA]
        ind.genes[indexA] = ind.genes[indexB]
        ind.genes[indexB] = tmp

        ind.computeFitness()
        
        self.updateBest(ind)
        return ind

    def updateMatingPool(self):
        """
        Updating the mating pool before creating a new generation
        """
        self.matingPool = []
        for ind_i in self.population:
            self.matingPool.append( ind_i.copy() )
        
            

    def newGeneration1(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            
            iA,iB=self.randomSelection()
            

#            ****Single Child Implementation**********
            child=self.uniformCrossover(iA,iB)
            childobj=self.createchildobject(child)
            self.population[i]=self.inversionMutation(childobj)
            
    def newGeneration2(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
#        

        for i in range(0, len(self.population)):
#            """
#            Depending of your experiment you need to use the most suitable algorithms for:
#            1. Select two candidates
#            2. Apply Crossover
#            3. Apply Mutation
#            """
            

            iA,iB=self.randomSelection()
            
            
    #            ****Single Child Implementation**********
            child=self.pmxCrossover(iA,iB)
            
            childobj=self.createchildobject(child)
            
            self.population[i]=self.mutation(childobj)
        
    
    def newGeneration3(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        
#        print("Printing Child after iteration:{}".format(self.iteration))
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            
            iA,iB=self.stochasticUniversalSampling()

            
#            ****Single Child Implementation**********
            child=self.uniformCrossover(iA,iB)
            childobj=self.createchildobject(child)
            self.population[i]=self.mutation(childobj)  
            
    def newGeneration4(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        

        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            
            iA,iB=self.stochasticUniversalSampling()

            
#            ****Single Child Implementation**********
            child=self.pmxCrossover(iA,iB)
            childobj=self.createchildobject(child)
            self.population[i]=self.mutation(childobj)
    def newGeneration5(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        
#        print("Printing Child after iteration:{}".format(self.iteration))
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            
            iA,iB=self.stochasticUniversalSampling()

            
#            ****Single Child Implementation**********
            child=self.pmxCrossover(iA,iB)
            childobj=self.createchildobject(child)
            self.population[i]=self.inversionMutation(childobj)
    def newGeneration6(self):
        """
        Creating a new generation
        1. Selection
        2. Crossover
        3. Mutation
        """
        
#        print("Printing Child after iteration:{}".format(self.iteration))
        for i in range(0, len(self.population)):
            """
            Depending of your experiment you need to use the most suitable algorithms for:
            1. Select two candidates
            2. Apply Crossover
            3. Apply Mutation
            """
            
            iA,iB=self.stochasticUniversalSampling()
#            iA,iB=self.randomSelection()
            
#            ****Single Child Implementation**********
            child=self.uniformCrossover(iA,iB)
            childobj=self.createchildobject(child)
            self.population[i]=self.inversionMutation(childobj)  
            
            
            
          
            
            
    def createchildobject(self,child):
        childobj=Individual(self.genSize, self.data)
        childobj.setGene(child)
        childobj.computeFitness()
        return childobj
            

    def GAStep(self,config):
        """
        One step in the GA main algorithm
        1. Updating mating pool with current population
        2. Creating a new Generation
        """

        self.updateMatingPool()
        if config==1:
            self.newGeneration1()
        elif config==2:
            self.newGeneration2()
        elif config==3:
            self.newGeneration3()
        elif config==4:
            self.newGeneration4()
        elif config==5:
            self.newGeneration5()
        elif config==6:
            self.newGeneration6()

    def search(self, config):
        """
        General search template.
        Iterates for a given number of steps
        """
        self.iteration = 0
        self.config=config
        while self.iteration < self.maxIterations and self.counter < 500:
            self.GAStep(self.config)
            self.iteration += 1
            self.counter+=1

        print ("Total iterations: ",self.iteration)
        print ("Best Solution: ", self.best.getFitness())

if len(sys.argv) < 2:
    print ("Error - Incorrect input")
    print ("Expecting python BasicTSP.py [instance] ")
    sys.exit(0)


problem_file = sys.argv[1]
population_size=sys.argv[2]
mutation_rate=sys.argv[3]
iterations=sys.argv[4]
configuration=int(sys.argv[5])



for i in range(5):
    print("Trial run{}:".format(i+1))
    print("Time:",datetime.datetime.now())
    ga = BasicTSP(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4],sys.argv[5])
    ga.search(configuration)