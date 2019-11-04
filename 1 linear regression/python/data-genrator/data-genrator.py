# -*- coding: utf-8 -*-
"""
Created on Wed Aug 14 22:57:45 2019

@author: M. Saqib
"""

# -*- coding: utf-8 -*-
"""
Created on Thu Aug  8 10:31:03 2019

@author: M. Saqib
"""
import random
fo = open("try2.txt", "w")
n=100
for i in range(n):
    st=str(i*5+random.randint(100,200))
    fo.write(st+"\n")
fo.close()

