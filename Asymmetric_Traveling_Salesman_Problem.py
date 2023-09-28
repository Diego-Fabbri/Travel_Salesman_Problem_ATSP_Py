import sys
import pandas as pd
import time, numpy as np
import pyomo.environ as pyo
from pyomo.environ import *
import math 
import random
import pandas as pd
import networkx as nx

random.seed(10) # Set seed to make results reproducible

#Parameters

n = 6 #Define Problem size

c = np.array([
                [np.inf, 33.6, 14, 40.9, 14.5, 11.5],
                 [34.7, np.inf, 21.7, 13, 20.2, 23.4],
                 [14.8, 21.5, np.inf, 29.3, 2, 3.9],
                 [41.7, 13.1, 29.4, np.inf, 27.6, 30.3],
                 [15, 20.2, 2, 27.5, np.inf, 3.9],
                 [12, 22.8, 2, 30.1, 4, np.inf]])

range_i = range(0,n)
range_j = range_i

#Create Model
model = pyo.ConcreteModel()

#Define variables
model.u = pyo.Var(range(0,n), # index i
                  bounds = (0,None),
                  initialize = 0)

model.x = pyo.Var(range(0,n), # index i
                  range(0,n), # index j
                  within = Binary,
                  initialize=0)

u = model.u
x = model.x

#Constraints 
model.C1 = pyo.ConstraintList() 
for j in range_j:
    model.C1.add(expr = sum(x[i,j] for i in range_i if i!= j)  == 1)
model.C2 = pyo.ConstraintList() 
for i in range_i:
    model.C2.add(expr = sum(x[i,j] for j in range_j if i!= j)  == 1)

model.C3 = pyo.ConstraintList() 
for i in range(1,n):
    for j in range(1,n):
        if i!= j:
            model.C3.add(expr = u[i] - u[j]+ n*x[i,j] <= n - 1)
            
# Define Objective Function
model.obj = pyo.Objective(expr = sum(c[i,j]*x[i,j] for i in range_i for j in range_j if i!= j), 
                          sense = minimize)

begin = time.time()
opt = SolverFactory('cplex')
results = opt.solve(model)

deltaT = time.time() - begin # Compute Exection Duration

model.pprint()

sys.stdout = open("Asymmetric_Travel_Salesman_ATSP_Problem_CLSP_Problem_Results.txt", "w") #Print Results on a .txt file

print('Time =', np.round(deltaT,2))

if (results.solver.status == SolverStatus.ok) and (results.solver.termination_condition == TerminationCondition.optimal):

    print('Total Cost (Obj value) =', pyo.value(model.obj))
    print('Solver Status is =', results.solver.status)
    print('Termination Condition is =', results.solver.termination_condition)
    print(" " )
    for i in range_i:
        for j in range_j:
            if  pyo.value(x[i,j]) != 0:
                print('x[' ,i, '][ ', j,']: ', round(pyo.value(x[i,j]),2))
    print(" " )
    for i in range_i:
        print('u[' ,i ,']: ', round(pyo.value(u[i]),2))
elif (results.solver.termination_condition == TerminationCondition.infeasible):
   print('Model is unfeasible')
  #print('Solver Status is =', results.solver.status)
   print('Termination Condition is =', results.solver.termination_condition)
else:
    # Something else is wrong
    print ('Solver Status: ',  result.solver.status)
    print('Termination Condition is =', results.solver.termination_condition)
    
sys.stdout.close()