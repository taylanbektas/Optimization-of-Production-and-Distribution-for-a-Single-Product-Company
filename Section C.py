#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import pandas as pd
from pyomo.environ import *
from itertools import chain

#Construct the model
model = ConcreteModel('Variable X_i_j')
# Define Sets
model.I = RangeSet(1,46)
model.J = RangeSet(1,46)

# Define Variable 
model.varX = Var(model.I, model.J, within = NonNegativeIntegers)


# In[13]:


######################################--------- CONSTRAINTS OF VARIABLES---------##############################################

########################             Q1               ########################

# Supply Constraint:
def constraint_S_Q1(model) :
    return model.varX[1,1] <= 7000
model.constraint_S_Q1 = Constraint(rule=constraint_S_Q1)

## from Procurement to Process constraint for Q1:

def constraint1(model, j) :
    return sum(model.varX[1, j] for j in range (2,6)) == model.varX[1,1]
model.constraint1 = Constraint(model.J, rule=constraint1)

## from Process to Assembly constraint for Q1:

def constraint2a(model, j):
    return sum(model.varX[2, j] for j in range(6, 8)) == model.varX[1, 2]
model.constraint2a = Constraint(model.J, rule=constraint2a)

def constraint2b(model, j):
    return model.varX[1, 2] <= 1800
model.constraint2b = Constraint(model.J, rule=constraint2b)

def constraint3a(model, j):
    return sum(model.varX[3, j] for j in range(6, 8)) == model.varX[1, 3]
model.constraint3a = Constraint(model.J, rule=constraint3a)

def constraint3b(model, j):
    return model.varX[1, 3] <= 1400
model.constraint3b = Constraint(model.J, rule=constraint3b)

def constraint4a(model, j):
    return sum(model.varX[4, j] for j in range(6, 8)) == model.varX[1, 4]
model.constraint4a = Constraint(model.J, rule=constraint4a)

def constraint4b(model, j):
    return model.varX[1, 4] <= 1900
model.constraint4b = Constraint(model.J, rule=constraint4b)

def constraint5a(model, j):
    return sum(model.varX[5, j] for j in range(6, 8)) == model.varX[1, 5]
model.constraint5a = Constraint(model.J, rule=constraint5a)

def constraint5b(model, j):
    return model.varX[1, 5] <= 1600
model.constraint5b = Constraint(model.J, rule=constraint5b)

#### ASSEMBLY EQUALITY

def constraint6a(model, i, j):
    return sum(model.varX[i, 6] for i in range(2, 6)) == sum(model.varX[6, j] for j in range(8, 12))
model.constraint6a = Constraint(model.I, model.J, rule=constraint6a)

def constraint6b(model, i):
    return sum(model.varX[i, 6] for i in range(2, 6)) <= 3450
model.constraint6b = Constraint(model.I, rule=constraint6b)

def constraint7a(model, i, j):
    return sum(model.varX[i, 7] for i in range(2, 6)) == sum(model.varX[7, j] for j in range(8, 12))
model.constraint7a = Constraint(model.I, model.J, rule=constraint7a)

def constraint7b(model, i):
    return sum(model.varX[i, 7] for i in range(2, 6)) <= 3200
model.constraint7b = Constraint(model.I, rule=constraint7b)

##### TESTING EQUALITY

def constraint8a(model, j):
    return sum(model.varX[8, j] for j in range(12, 16)) == model.varX[6, 8] + model.varX[7, 8]
model.constraint8a = Constraint(model.J, rule=constraint8a)

def constraint8b(model, j):
    return model.varX[6, 8] + model.varX[7, 8] <= 1750
model.constraint8b = Constraint(model.J, rule=constraint8b)

def constraint9a(model, j):
    return sum(model.varX[9, j] for j in range (12,16)) == model.varX[6, 9]+ model.varX[7,9]
model.constraint9a = Constraint(model.J, rule=constraint9a)

def constraint9b(model, j):
    return model.varX[6, 9]+ model.varX[7,9] <= 1550
model.constraint9b = Constraint(model.J, rule=constraint9b)


def constraint10a(model, j):
    return sum(model.varX[10, j] for j in range (12,16)) == model.varX[6, 10]+ model.varX[7,10]
model.constraint10a = Constraint(model.J, rule=constraint10a)

def constraint10b(model, j):
    return model.varX[6, 10]+ model.varX[7,10] <= 1800
model.constraint10b = Constraint(model.J, rule=constraint10b)


def constraint11a(model, j):
    return sum(model.varX[11, j] for j in range (12,16)) == model.varX[6, 11] + model.varX[7,11]
model.constraint11a = Constraint(model.J, rule=constraint11a)

def constraint11b(model, j):
    return model.varX[6, 11] + model.varX[7,11] <= 1450
model.constraint11b = Constraint(model.J, rule=constraint11b)


## from Packaging nodes to Inventory nodes constraints:

def constraint12a(model, i, j):
    return sum(model.varX[i, 12] for i in range(8,12)) == sum(model.varX[12,j] for j in (31,35,36,37,38))
model.constraint12a = Constraint(model.I, model.J, rule=constraint12a)

def constraint12b(model, i, j):
    return sum(model.varX[i, 12] for i in range (8,12)) <= 1700
model.constraint12b = Constraint(model.I, model.J, rule=constraint12b)


def constraint13a(model, i, j):
    return sum(model.varX[i, 13] for i in range (8,12)) == sum(model.varX[13,j] for j in (32,35,36,37,38))
model.constraint13a = Constraint(model.I, model.J, rule=constraint13a)

def constraint13b(model, i, j):
    return sum(model.varX[i, 13] for i in range (8,12)) <= 1200
model.constraint13b = Constraint(model.I, model.J, rule=constraint13b)


def constraint14a(model, i, j):
    return sum(model.varX[i, 14] for i in range (8,12)) == sum(model.varX[14,j] for j in (33,35,36,37,38))
model.constraint14a = Constraint(model.I, model.J, rule=constraint14a)

def constraint14b(model, i, j):
    return sum(model.varX[i, 14] for i in range (8,12)) <= 2000
model.constraint14b = Constraint(model.I, model.J, rule=constraint14b)


def constraint15a(model, i, j):
    return sum(model.varX[i, 15] for i in range (8,12)) == sum(model.varX[15,j] for j in range(34,39))

def constraint15b(model, i):
    return sum(model.varX[i, 15] for i in range (8,12)) <= 1400

model.constraint15a = Constraint(model.I, model.J, rule=constraint15a)
model.constraint15b = Constraint(model.I, rule=constraint15b)


# from Inventory to Warehouse to sort :

def constraint16a(model, j):
    return model.varX[12, 31]  == sum(model.varX[31,j] for j in range(39,41))

def constraint16b(model):
    return model.varX[12, 31]  <= 100

model.constraint16a = Constraint(model.J, rule=constraint16a)
model.constraint16b = Constraint(rule=constraint16b)


def constraint17a(model, j):
    return model.varX[13, 32] == sum(model.varX[32,j] for j in range(39,41))

def constraint17b(model):
    return model.varX[13, 32] <= 80 

model.constraint17a = Constraint(model.J, rule=constraint17a)
model.constraint17b = Constraint(rule=constraint17b)


def constraint18a(model, j):
    return model.varX[14, 33] == sum(model.varX[33,j] for j in range(39,41))

def constraint18b(model):
    return model.varX[14, 33] <= 100

model.constraint18a = Constraint(model.J, rule=constraint18a)
model.constraint18b = Constraint(rule=constraint18b)


def constraint19a(model, j):
    return model.varX[15, 34] == sum(model.varX[34,j] for j in range(39,41))

def constraint19b(model):
    return model.varX[15, 34] <= 90 

model.constraint19a = Constraint(model.J, rule=constraint19a)
model.constraint19b = Constraint(rule=constraint19b)


def constraint20a(model, i, j):
    return sum(model.varX[i, 35] for i in range(12,16)) == model.varX[35,39]

def constraint20b(model, i):
    return sum(model.varX[i, 35] for i in range(12,16)) <= 2700

model.constraint20a = Constraint(model.I, model.J, rule=constraint20a)
model.constraint20b = Constraint(model.I, rule=constraint20b)

def constraint21a(model, i, j) :
    # Kayseri Inventory capacity
    return sum(model.varX[i, 36] for i in range(12,16)) == model.varX[36,40]
model.constraint21a = Constraint(model.I, model.J, rule=constraint21a)

def constraint21b(model, i, j) :
    # Kayseri Inventory capacity
    return sum(model.varX[i, 36] for i in range(12,16)) <= 2000
model.constraint21b = Constraint(model.I, model.J, rule=constraint21b)


def constraint22a(model, i, j) : # İzmir Warehouse sorting capacity at Q1
    return sum(model.varX[i, 37] for i in range (12,16)) == sum(model.varX[37,j] for j in range(41,44))
model.constraint22a = Constraint(model.I, model.J, rule=constraint22a)

def constraint22b(model, i, j) : # İzmir Warehouse sorting capacity at Q1
    return sum(model.varX[i, 37] for i in range (12,16)) <= 3500
model.constraint22b = Constraint(model.I, model.J, rule=constraint22b)


def constraint23a(model, i, j) : # Kayseri Warehouse sorting capacity at Q1
    return sum(model.varX[i, 38] for i in range (12,16)) == sum(model.varX[38,j] for j in range(41,44))
model.constraint23a = Constraint(model.I, model.J, rule=constraint23a)

def constraint23b(model, i, j) : # Kayseri Warehouse sorting capacity at Q1
    return sum(model.varX[i, 38] for i in range (12,16)) <= 2800
model.constraint23b = Constraint(model.I, model.J, rule=constraint23b)


def constraint24a(model, i, j) : # İzmir Warehouse sorting capacity at Q2
    return sum(model.varX[i, 39] for i in range(31,36)) + sum(model.varX[i, 39] for i in range(27,31)) == sum(model.varX[39,j] for j in range (41,47))
model.constraint24a = Constraint(model.I, model.J, rule=constraint24a)

def constraint24b(model, i, j) : # İzmir Warehouse sorting capacity at Q2
    return sum(model.varX[i, 39] for i in range(31,36)) + sum(model.varX[i, 39] for i in range (27,31)) <= 3500
model.constraint24b = Constraint(model.I, model.J, rule=constraint24b)

def constraint25a(model, i, j):
    # Kayseri Warehouse sorting capacity at Q2
    return sum(model.varX[i, 40] for i in (31,32,33,34,36)) + sum(model.varX[i, 40] for i in range(27, 31)) == sum(model.varX[40, j] for j in range(41, 47))

model.constraint25a = Constraint(model.I, model.J, rule=constraint25a)

def constraint25b(model, i, j):
    # Kayseri Warehouse sorting capacity at Q2
    return sum(model.varX[i, 40] for i in (31,32,33,34,36)) + sum(model.varX[i, 40] for i in range(27, 31)) <= 2800

model.constraint25b = Constraint(model.I, model.J, rule=constraint25b)

# Demand Constraints: constraint_D_T_1 means that constraint for demand of Tepco at Quarter1 
def constraint_D_T_1(model) :
    return sum(model.varX[i,41] for i in range (37,41)) == 2000
model.constraint_D_T_1 = Constraint(rule=constraint_D_T_1)

def constraint_D_T_2(model) :
    return sum(model.varX[i,44] for i in range (39,41)) == 2500
model.constraint_D_T_2 = Constraint(rule=constraint_D_T_2)

def constraint_D_C_1(model) :
    return sum(model.varX[i,42] for i in range (37,41)) == 1700
model.constraint_D_C_1 = Constraint(rule=constraint_D_C_1)

def constraint_D_C_2(model) :
    return sum(model.varX[i,45] for i in range (39,41)) == 1500
model.constraint_D_C_2 = Constraint(rule=constraint_D_C_2)

def constraint_D_B_1(model) :
    return sum(model.varX[i,43] for i in range (37,41)) == 1800
model.constraint_D_B_1 = Constraint(rule=constraint_D_B_1)

def constraint_D_B_2(model) :
    return sum(model.varX[i,46] for i in range (39,41)) == 2050
model.constraint_D_B_2 = Constraint(rule=constraint_D_B_2)


# In[14]:


############################        Q2       #################################

# Supply Constraint:
def constraint_S_Q2(model) :
    return model.varX[16,16] <= 7000
model.constraint_S_Q2 = Constraint(rule=constraint_S_Q2)

## from Procurement to Process constraint:

def constraint26(model, j) :
    return sum(model.varX[16, j] for j in range(17,21)) == model.varX[16,16]
model.constraint26 = Constraint(model.J, rule=constraint26)

## from Process to Assembly constraint:

def constraint27a(model, j):
    return sum(model.varX[17, j] for j in range(21, 23)) == model.varX[16, 17]

def constraint27b(model, j):
    return model.varX[16, 17] <= 1800

model.constraint27a = Constraint(model.J, rule=constraint27a)
model.constraint27b = Constraint(model.J, rule=constraint27b)

def constraint28a(model, j):
    return sum(model.varX[18, j] for j in range(21, 23)) == model.varX[16, 18]

def constraint28b(model, j):
    return model.varX[16, 18] <= 1400

model.constraint28a = Constraint(model.J, rule=constraint28a)
model.constraint28b = Constraint(model.J, rule=constraint28b)

def constraint29b(model, j):
    return model.varX[16, 19] <= 1900
model.constraint29b = Constraint(model.J, rule=constraint29b)
def constraint29a(model, j):
    return sum(model.varX[19, j] for j in range(21, 23)) == model.varX[16, 19]
model.constraint29a = Constraint(model.J, rule=constraint29a)


def constraint30a(model, j):
    return sum(model.varX[20, j] for j in range(21, 23)) == model.varX[16, 20]

def constraint30b(model, j):
    return model.varX[16, 20] <= 1600

model.constraint30a = Constraint(model.J, rule=constraint30a)
model.constraint30b = Constraint(model.J, rule=constraint30b)

def constraint31a(model, i, j):
    return sum(model.varX[i, 21] for i in range(17, 21)) == sum(model.varX[21, j] for j in range(23, 27))

def constraint31b(model, i):
    return sum(model.varX[i, 21] for i in range(17, 21)) <= 3450

model.constraint31a = Constraint(model.I, model.J, rule=constraint31a)
model.constraint31b = Constraint(model.I, rule=constraint31b)

def constraint32a(model, i, j):
    return sum(model.varX[i, 22] for i in range(17, 21)) == sum(model.varX[22, j] for j in range(23, 27))

def constraint32b(model, i):
    return sum(model.varX[i, 22] for i in range(17, 21)) <= 3200

model.constraint32a = Constraint(model.I, model.J, rule=constraint32a)
model.constraint32b = Constraint(model.I, rule=constraint32b)

def constraint33a(model, j):
    return sum(model.varX[23, j] for j in range(27, 31)) == model.varX[21, 23] + model.varX[22, 23]

def constraint33b(model, j):
    return model.varX[21, 23] + model.varX[22, 23] <= 1750

model.constraint33a = Constraint(model.J, rule=constraint33a)
model.constraint33b = Constraint(model.J, rule=constraint33b)

def constraint34a(model, j):
    return sum(model.varX[24, j] for j in range(27,31)) == model.varX[21, 24]+ model.varX[22,24]
model.constraint34a = Constraint(model.J, rule=constraint34a)

def constraint34b(model, j):
    return model.varX[21, 24]+ model.varX[22,24] <= 1550
model.constraint34b = Constraint(model.J, rule=constraint34b)

def constraint35a(model, j):
    return sum(model.varX[25, j] for j in range(27,31)) == model.varX[21, 25]+ model.varX[22,25]
model.constraint35a = Constraint(model.J, rule=constraint35a)

def constraint35b(model, j):
    return model.varX[21, 25]+ model.varX[22,25] <= 1800
model.constraint35b = Constraint(model.J, rule=constraint35b)

def constraint36a(model, j):
    return sum(model.varX[26, j] for j in range(27,31)) == model.varX[21, 26] + model.varX[22,26]
model.constraint36a = Constraint(model.J, rule=constraint36a)

def constraint36b(model, j):
    return model.varX[21, 26] + model.varX[22,26] <= 1450
model.constraint36b = Constraint(model.J, rule=constraint36b)


#### Package Q2 STAGE
def constraint37a(model, i):
    return sum(model.varX[i, 27] for i in range(23,27)) == model.varX[27, 39] + model.varX[27,40]
model.constraint37a = Constraint(model.J, rule=constraint37a)

def constraint37b(model, i):
    return sum(model.varX[i, 27] for i in range(23,27)) <= 1700
model.constraint37b = Constraint(model.J, rule=constraint37b)

def constraint38a(model, i):
    return sum(model.varX[i, 28] for i in range (23,27)) == model.varX[28, 39] + model.varX[28,40]
model.constraint38a = Constraint(model.J, rule=constraint38a)

def constraint38b(model, i):
    return sum(model.varX[i, 28] for i in range(23,27)) <= 1200
model.constraint38b = Constraint(model.J, rule=constraint38b)

def constraint39a(model, i):
    return sum(model.varX[i, 29] for i in range(23,27)) == model.varX[29, 39] + model.varX[29,40]
model.constraint39a = Constraint(model.J, rule=constraint39a)

def constraint39b(model, i):
    return sum(model.varX[i, 29] for i in range(23,27)) <= 2000
model.constraint39b = Constraint(model.J, rule=constraint39b)

def constraint40a(model, i):
    return sum(model.varX[i, 30] for i in range(23, 27)) == model.varX[30, 39] + model.varX[30, 40]
model.constraint40a = Constraint(model.J, rule=constraint40a)

def constraint40b(model, i):
    return sum(model.varX[i, 30] for i in range(23, 27)) <= 1400
model.constraint40b = Constraint(model.J, rule=constraint40b)

# from sorting to retailer and from inventory to soring then to retailer

def constraint41(model, i, j) :
    return sum(model.varX[i,37] for i in range(12,16)) == sum(model.varX[37,j] for j in range(41,44))
model.constraint41 = Constraint(model.I, model.J, rule=constraint41)    

def constraint42(model, i, j) :
    return sum(model.varX[i,38] for i in range(12,16)) == sum(model.varX[38,j] for j in range(41,44))
model.constraint42 = Constraint(model.I, model.J, rule=constraint42)      
    
def constraint43(model, i, j) :
    return sum(model.varX[i,39] for i in range(31,36)) + sum(model.varX[i,39] for i in range(27,31) ) == sum(model.varX[39,j] for j in range(41,47))
model.constraint43 = Constraint(model.I, model.J, rule=constraint43)  
    
def constraint44(model, i, j) :
    return sum(model.varX[i,40] for i in (31,32,33,34,36)) + sum(model.varX[i,40] for i in range(27,31) ) == sum(model.varX[40,j] for j in range(41,47))
model.constraint44 = Constraint(model.I, model.J, rule=constraint44)      


# In[15]:


###Parameters for transportation per product stage

ank_esk = 235
ank_ist = 445
ank_ant = 476
esk_ist = 303
esk_ant = 415
ist_ant = 695

ank_izm = 589
ank_kay = 346
esk_izm = 417
esk_kay = 512
ist_izm = 479
ist_kay = 774
ant_izm = 454
ant_kay = 573

izm_tep = 202
izm_car = 561
izm_big = 867
kay_tep = 817
kay_car = 304
kay_big = 453

raw = 2
processed = 1.75
ass = 1.5
test = 1.5
packaged = 1.5
sort = 1.5

### Cost Matrices ####

##################----------------Distance times transportation cost of production stage------------------------########## 
M = 100000

def times(a,b):
    return a*b

x = np.array([2, 1.75, 1.5, 1.5, 1.5, 1.5])
y = np.array([235, 445, 476, 303, 415, 695, 589, 346, 417, 512, 479, 774, 454 ,573, 202, 561, 867, 817, 304, 453])

cost_matrix_distance = np.zeros((x.shape[0], y.shape[0]))
c = cost_matrix_distance
# Fill in the cost matrix with if statements
for i in range(x.shape[0]):
    for j in range(y.shape[0]):
        if i == 0 and j >= 3:
            cost_matrix_distance[i, j] = M
        elif (i == 1 or i == 2) and (j >= 6 or j == 0):
            cost_matrix_distance[i, j] = M
        elif i == 3 and (j>= 6):
            cost_matrix_distance[i, j] = M
        elif i == 4 and j <= 6:
            cost_matrix_distance[i, j] = M
        elif i == 5 and j <= 13:
            cost_matrix_distance[i, j] = M
        else:
            cost_matrix_distance[i, j] = times(x[i],y[j])
            

############# Cost of production stage ##################

raw_ank = 35
process_ank = 56
process_esk = 42
process_ist = 35
process_ant = 49
ass_ist = 21
ass_ant = 35
test_ank = 28
test_esk = 35
test_ist = 21
test_ant = 42
pack_ank = 14
pack_esk = 21
pack_ist = 28
pack_ant = 35
sort_izm = 15
sort_kay = 12

invent_ank = 15
invent_esk = 12
invent_ist = 20
invent_ant = 19
invent_izm = 7
invent_kay = 5



# In[16]:


def proc_ank(model):
    return raw_ank*model.varX[1,1]

def process(model):
    result1 = 0
    for i in range(1,2) :
        for j in range(2,6):
            if j == 2:
                result1 += process_ank*model.varX[i,j]
            elif j == 3:
                result1 += process_esk*model.varX[i,j]+(ank_esk*raw*model.varX[i,j])
            elif j== 4:
                result1 += process_ist*model.varX[i,j]+(ank_ist*raw*model.varX[i,j])
            elif j==5:
                result1 += process_ant*model.varX[i,j]+(ank_ant*raw*model.varX[i,j])
    return result1
            
def assembly(model):
    result2 = 0
    for i in range(2,6) :
        for j in (6,7):
            if i== 2 and j == 6:
                result2 += ass_ist*model.varX[i,j]+(ank_ist * processed * model.varX[i,j])
            elif i== 2 and j == 7:
                result2 += ass_ant*model.varX[i,j]+(ank_ant * processed * model.varX[i,j])
            elif i== 3 and j == 6:
                result2 += ass_ist*model.varX[i,j]+(esk_ist*processed*model.varX[i,j])
            elif i== 3 and j == 7:
                result2 += ass_ant*model.varX[i,j]+(esk_ant*processed*model.varX[i,j])
            elif i== 4 and j == 6:
                result2 += ass_ist*model.varX[i,j]
            elif i== 4 and j == 7:
                result2 += ass_ant*model.varX[i,j]+(ist_ant*processed*model.varX[i,j])
            elif i== 5 and j == 6:
                result2 += ass_ist*model.varX[i,j]+(ist_ant*processed*model.varX[i,j])
            elif i== 5 and j == 7:
                result2 += ass_ant*model.varX[i,j]
    return result2 
            
def testing(model):
    result3 = 0
    for i in (6,7):
        for j in range(8,12):
            if i == 6 and j == 8:
                result3 += test_ank * model.varX[i,j] + ank_ist*ass * model.varX[i,j]
            elif i == 6 and j == 9:
                result3 += test_esk * model.varX[i,j] + esk_ist*ass* model.varX[i,j]
            elif i == 6 and j == 10:
                result3 += test_ist * model.varX[i,j]
            elif i == 6 and j == 11:
                result3 += test_ant * model.varX[i,j] + ist_ant*ass* model.varX[i,j]
            elif i == 7 and j == 8:
                result3 += test_ank * model.varX[i,j] + ank_ant*ass * model.varX[i,j]
            elif i == 7 and j == 9:
                result3 += test_esk * model.varX[i,j] + esk_ant*ass * model.varX[i,j]
            elif i == 7 and j == 10:
                result3 += test_ist * model.varX[i,j] + ist_ant*ass * model.varX[i,j]
            elif i == 7 and j == 11:
                result3 += test_ant * model.varX[i,j]
    return result3

            

def package(model):
    result4 = 0
    for i in range(8,12) :
        for j in range(12,16):
            if i == 8 and j == 12:
                result4 += pack_ank*model.varX[i,j]
            elif i == 8 and j == 13:
                result4 += pack_esk*model.varX[i,j]+(c[3,0]*model.varX[i,j])
            elif i == 8 and j == 14:
                result4 += pack_ist*model.varX[i,j]+(c[3,1]*model.varX[i,j])
            elif i == 8 and j == 15:
                result4 += pack_ant*model.varX[i,j]+(c[3,2]*model.varX[i,j])
            elif i == 9 and j == 12:
                result4 += pack_ank*model.varX[i,j]+(c[3,0]*model.varX[i,j])
            elif i == 9 and j == 13:
                result4 += pack_esk*model.varX[i,j]
            elif i == 9 and j == 14:
                result4 += pack_ist*model.varX[i,j]+(c[3,3]*model.varX[i,j])
            elif i == 9 and j == 15:
                result4 += pack_ant*model.varX[i,j]+(c[3,4]*model.varX[i,j])
            elif i == 10 and j == 12:
                result4 += pack_ank*model.varX[i,j]+(c[3,2]*model.varX[i,j])
            elif i == 10 and j == 13:
                result4 += pack_esk*model.varX[i,j]+(c[3,3]*model.varX[i,j])
            elif i == 10 and j == 14:
                result4 += pack_ist*model.varX[i,j]
            elif i == 10 and j == 15:
                result4 += pack_ant*model.varX[i,j]+(c[3,5]*model.varX[i,j])
            elif i == 11 and j == 12:
                result4 += pack_ank*model.varX[i,j]+(c[3,2]*model.varX[i,j])
            elif i == 11 and j == 13:
                result4 += pack_esk*model.varX[i,j]+(c[3,4]*model.varX[i,j])
            elif i == 11 and j == 14:
                result4 += pack_ist*model.varX[i,j]+(c[3,5]*model.varX[i,j])
            elif i == 11 and j == 15:
                result4 += pack_ant*model.varX[i,j]
    return result4 
            
        
            
def sortQ1(model):
    result5 = 0
    for i in range(12,16) :
        for j in (37,38):
            if i == 12 and j == 37:
                result5 += sort_izm*model.varX[i,j]+(c[4,6]*model.varX[i,j])
            elif i == 12 and j==38:
                result5 += sort_kay*model.varX[i,j]+(c[4,7]*model.varX[i,j])
            elif i == 13 and j == 37:
                result5 += sort_izm*model.varX[i,j]+(c[4,8]*model.varX[i,j])
            elif i == 13 and j== 38:
                result5 += sort_kay*model.varX[i,j]+(c[4,9]*model.varX[i,j])
            elif i == 14 and j == 37:
                result5 += sort_izm*model.varX[i,j]+(c[4,10]*model.varX[i,j])
            elif i == 14 and j== 38:
                result5 += sort_kay*model.varX[i,j]+(c[4,11]*model.varX[i,j])
            elif i == 15 and j == 37:
                result5 += sort_izm*model.varX[i,j]+(c[4,12]*model.varX[i,j])
            elif i == 15 and j == 38:
                result5 += sort_kay*model.varX[i,j]+(c[4,13]*model.varX[i,j])
    return result5
            
            
def sortQ2_1(model):
    result = 0
    for i in range(27, 31):
        for j in (39,40):
            if i == 27 and j == 39:
                result += sort_izm*model.varX[i,j]+(ank_izm*packaged*1.4*model.varX[i,j])
            elif i == 27 and j== 40:
                result += sort_kay*model.varX[i,j]+(ank_kay*packaged*1.4*model.varX[i,j])
            elif i == 28 and j == 39:
                result += sort_izm*model.varX[i,j]+(esk_izm*packaged*1.4*model.varX[i,j])
            elif i == 28 and j== 40:
                result += sort_kay*model.varX[i,j]+(esk_kay*packaged*1.4*model.varX[i,j])
            elif i == 29 and j == 39:
                result += sort_izm*model.varX[i,j]+(ist_izm*packaged*1.4*model.varX[i,j])
            elif i == 29 and j== 40:
                result += sort_kay*model.varX[i,j]+(ist_kay*packaged*1.4*model.varX[i,j])
            elif i == 30 and j == 39:
                result += sort_izm*model.varX[i,j]+(ant_izm*packaged*1.4*model.varX[i,j])
            elif i == 30 and j == 40:
                result += sort_kay*model.varX[i,j]+(ant_kay*packaged*1.4*model.varX[i,j])
    return result
                
def sortQ2_2(model):
    result0 = 0
    for i in range(31,37):
        for j in (39,40):
            if i == 31 and j == 39:
                result0 += sort_izm*model.varX[i,j]+(ank_izm*packaged*1.4*model.varX[i,j])
            elif i == 31 and j==40:
                result0 += sort_kay*model.varX[i,j]+(ank_kay*packaged*1.4*model.varX[i,j])
            elif i == 32 and j == 39:
                result0 += sort_izm*model.varX[i,j]+(esk_izm*packaged*1.4*model.varX[i,j])
            elif i == 32 and j==40:
                result0 += sort_kay*model.varX[i,j]+(esk_ant*packaged*1.4*model.varX[i,j])
            elif i == 33 and j == 39:
                result0 += sort_izm*model.varX[i,j]+(ist_izm*packaged*1.4*model.varX[i,j])
            elif i == 33 and j==40:
                result0 += sort_kay*model.varX[i,j]+(c[4,11]*1.4*model.varX[i,j])
            elif i == 34 and j == 39:
                result0 += sort_izm*model.varX[i,j]+(c[4,12]*1.4*model.varX[i,j])
            elif i == 34 and j==40:
                result0 += sort_kay*model.varX[i,j]+(c[4,13]*1.4*model.varX[i,j])
            elif i == 35 and j == 39:
                result0 += sort_izm*model.varX[i,j]
            elif i == 35 and j== 40:
                result0 += M*M*M*model.varX[i,j]
            elif i == 36 and j == 39:
                result0 += M*M*M*model.varX[i,j]
            elif i == 36 and j == 40:
                result0 += sort_kay*model.varX[i,j]
    return result0
            
            
def inventory(model):
    result6 = 0
    for i in range(12,16) :
        for j in range(31,37):
            if i==12 and j == 31:
                result6 += invent_ank*model.varX[i,j]
            elif i == 13 and j == 32:
                result6 += invent_esk*model.varX[i,j]
            elif i == 14 and j== 33:
                result6 += invent_ist*model.varX[i,j]
            elif i == 15 and j== 34:
                result6 += invent_ant*model.varX[i,j]
            elif i == 12 and j== 35:
                result6 += invent_izm*model.varX[i,j]+(ank_izm*packaged*model.varX[i,j])
            elif i == 13 and j== 35:
                result6 += invent_izm*model.varX[i,j]+(esk_izm*packaged*model.varX[i,j]) 
            elif i == 14 and j== 35:
                result6 += invent_izm*model.varX[i,j]+(ist_izm*packaged*model.varX[i,j])
            elif i == 15 and j== 35:
                result6 += invent_izm*model.varX[i,j]+(ant_izm*packaged*model.varX[i,j])
            elif i == 12 and j== 36:
                result6 += invent_kay*model.varX[i,j]+(ank_kay*packaged*model.varX[i,j])
            elif i == 13 and j== 36:
                result6 += invent_kay*model.varX[i,j]+(esk_kay*packaged*model.varX[i,j])
            elif i == 14 and j== 36:
                result6 += invent_kay*model.varX[i,j]+(ist_kay*packaged*model.varX[i,j])
            elif i == 15 and j== 36:
                result6 += invent_kay*model.varX[i,j]+(ant_kay*packaged*model.varX[i,j])
    return result6
            
            
def retailer(model):
    result7 = 0
    for i in range(37,39):
        for j in range(41,44):
            if i == 37 and j == 41:
                result7 += izm_tep*sort*model.varX[i,j]
            elif i == 37 and j == 42:
                result7 += izm_car*sort*model.varX[i,j]
            elif i == 37 and j == 43:
                result7 += izm_big*sort*model.varX[i,j]
            elif i == 38 and j == 41:
                result7 += kay_tep*sort*model.varX[i,j]
            elif i == 38 and j == 42:
                result7 += kay_car*sort*model.varX[i,j]
            elif i == 38 and j == 43:
                result7 += kay_big*sort*model.varX[i,j]
    return result7 
            

            
def retailerQ2(model):
    result8 = 0
    for i in range(39,41):
        for j in range(41,47):
            if i == 39 and j == 41:
                result8 += izm_tep*sort*1.4*model.varX[i,j]+(model.varX[i,j]*37)
            elif i == 39 and j == 42:
                result8 += izm_car*sort*1.4*model.varX[i,j]+(model.varX[i,j]*37)
            elif i == 39 and j == 43:
                result8 += izm_big*sort*1.4*model.varX[i,j]+(model.varX[i,j]*37)
            elif i == 39 and j == 44:
                result8 += izm_tep*sort*1.4*model.varX[i,j]
            elif i == 39 and j == 45:
                result8 += izm_car*sort*1.4*model.varX[i,j]
            elif i == 39 and j == 46:
                result8 += izm_big*sort*1.4*model.varX[i,j]
            elif i == 40 and j == 41:
                result8 += c[5,17]*1.4*model.varX[i,j] + (model.varX[i,j]*37)
            elif i == 40 and j == 42:
                result8 += c[5,18]*1.4*model.varX[i,j] + (model.varX[i,j]*37)
            elif i == 40 and j == 43:
                result8 += c[5,19]*1.4*model.varX[i,j] + (model.varX[i,j]*37)
            elif i == 40 and j == 44:
                result8 += c[5,17]*1.4*model.varX[i,j]
            elif i == 40 and j == 45:
                result8 += c[5,18]*1.4*model.varX[i,j]
            elif i == 40 and j == 46:
                result8 += c[5,19]*1.4*model.varX[i,j]
    return result8 
 


# In[17]:


### for Quarter 2 ###

def proc_ankQ2(model):
    return raw_ank*model.varX[16,16]

def processQ2(model):
    result9 = 0
    for i in range(16,17) :
        for j in range(17,21):
            if j == 17:
                result9 += process_ank*model.varX[i,j]
            elif j == 18:
                result9 += process_esk*model.varX[i,j]+(ank_esk*raw*1.4*model.varX[i,j])
            elif j== 19:
                result9 += process_ist*model.varX[i,j]+(ank_ist*raw*1.4*model.varX[i,j])
            elif j== 20:
                result9 += process_ant*model.varX[i,j]+(ank_ant*raw*1.4*model.varX[i,j])
    return result9
                
            
            
def assemblyQ2(model):
    result10 = 0
    for i in range(17,21) :
        for j in (21,22):
            if i== 17 and j == 21:
                result10 += ass_ist*model.varX[i,j]+(ank_ist*processed*1.4*model.varX[i,j])
            elif i== 17 and j == 22:
                result10 += ass_ant*model.varX[i,j]+(ank_ant*processed*1.4*model.varX[i,j])
            elif i== 18 and j == 21:
                result10 += ass_ist*model.varX[i,j]+(esk_ist*processed*1.4*model.varX[i,j])
            elif i== 18 and j == 22:
                result10 += ass_ant*model.varX[i,j]+(esk_ant*processed*1.4*model.varX[i,j])
            elif i== 19 and j == 21:
                result10 += ass_ist*model.varX[i,j]
            elif i== 19 and j == 22:
                result10 += ass_ant*model.varX[i,j]+(ist_ant*processed*1.4*model.varX[i,j])
            elif i== 20 and j == 21:
                result10 += ass_ist*model.varX[i,j]+(ist_ant*processed*1.4*model.varX[i,j])
            elif i== 20 and j == 22:
                result10 += ass_ant*model.varX[i,j]
    return result10 
            
            
def testingQ2(model):
    result11 = 0
    for i in (21,22) :
        for j in range(23,27):
            if i== 21 and j == 23:
                result11 += test_ank*model.varX[i,j]+(ank_ist*ass*1.4*model.varX[i,j])
            elif i == 21 and j == 24:
                result11 += test_esk*model.varX[i,j]+(esk_ist*ass*1.4*model.varX[i,j])
            elif i == 21 and j == 25:
                result11 += test_ist*model.varX[i,j]
            elif i == 21 and j == 26:
                result11 += test_ant*model.varX[i,j]+(ist_ant*ass*1.4*model.varX[i,j])
            elif i == 22 and j == 23:
                result11 += test_ank*model.varX[i,j]+(ank_ant*ass*1.4*model.varX[i,j])
            elif i == 22 and j == 24:
                result11 += test_esk*model.varX[i,j]+(esk_ant*ass*1.4*model.varX[i,j])
            elif i == 22 and j == 25:
                result11 += test_ist*model.varX[i,j]+(ist_ant*ass*1.4*model.varX[i,j])
            elif i == 22 and j == 26:
                result11 += test_ant*model.varX[i,j]
    return result11 
            
def packageQ2(model):
    result12 = 0
    for i in range(23,27) :
        for j in range(27,31):
            if i == 23 and j ==27:
                result12 += pack_ank*model.varX[i,j]
            elif i == 23 and j == 28:
                result12 += pack_esk*model.varX[i,j]+(ank_esk*test*1.4*model.varX[i,j])
            elif i == 23 and j == 29:
                result12 += pack_ist*model.varX[i,j]+(ank_ist*test*1.4*model.varX[i,j])
            elif i == 23 and j == 30:
                result12 += pack_ant*model.varX[i,j]+(ank_ant*test*1.4*model.varX[i,j])
            elif i == 24 and j == 27:
                result12 += pack_ank*model.varX[i,j]+(ank_esk*test*1.4*model.varX[i,j])
            elif i == 24 and j == 28:
                result12 += pack_esk*model.varX[i,j]
            elif i == 24 and j == 29:
                result12 += pack_ist*model.varX[i,j]+(esk_ist*test*1.4*model.varX[i,j])
            elif i == 24 and j == 30:
                result12 += pack_ant*model.varX[i,j]+(esk_ant*test*1.4*model.varX[i,j])
            elif i == 25 and j == 27:
                result12 += pack_ank*model.varX[i,j]+(ank_ist*test*1.4*model.varX[i,j])
            elif i == 25 and j == 28:
                result12 += pack_esk*model.varX[i,j]+(esk_ist*test*1.4*model.varX[i,j])
            elif i == 25 and j == 29:
                result12 += pack_ist*model.varX[i,j]
            elif i == 25 and j == 30:
                result12 += pack_ant*model.varX[i,j]+(ist_ant*test*1.4*model.varX[i,j])
            elif i == 26 and j == 27:
                result12 += pack_ank*model.varX[i,j]+(ank_ant*test*1.4*model.varX[i,j])
            elif i == 26 and j == 28:
                result12 += pack_esk*model.varX[i,j]+(esk_ant*test*1.4*model.varX[i,j])
            elif i == 26 and j == 29:
                result12 += pack_ist*model.varX[i,j]+(ist_ant*test*1.4*model.varX[i,j])
            elif i == 26 and j == 30:
                result12 += pack_ant*model.varX[i,j]
    return result12 


# In[18]:


def obj_rule(model):
    return  proc_ank(model) + proc_ankQ2(model) + process(model) + processQ2(model) + assembly(model) + assemblyQ2(model) + testing(model) + testingQ2(model) + package(model) + packageQ2(model) + sortQ1(model) + sortQ2_1(model) + sortQ2_2(model) + inventory(model) + retailer(model) + retailerQ2(model)
model.obj = Objective(rule=obj_rule, sense=minimize)

# solve the model first
solver = SolverFactory('glpk')
solver.solve(model)

# print the value of the objective function
print(f"Objective value: {model.obj()}")

# NUMBER OF BACK-ORDER
print(f"Number of back-orders: {model.varX[39,41].value + model.varX[39,42].value + model.varX[39,43].value + model.varX[40,41].value + model.varX[40,42].value + model.varX[40,43].value}")

