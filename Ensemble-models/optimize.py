import scipy.optimize as opt
from   scipy.optimize import rosen,rosen_der
import numpy as np
import math
import matplotlib.pyplot as plt
import sys
import gzip
import pandas as pd
from sklearn.metrics import roc_auc_score as ras

#### Input format: ####
#### python3 *.py nin in1 in2 ....####

### Read-In All Training Data && Testing Data ###
def readcsv(iostr):
  with gzip.open(iostr,'r') as f:
       str=pd.read_csv(f)
  return str

nin=int(sys.argv[1])
if(nin>10):
  print("maximum input files is %d" % nin)
  print("exiting")
  exit(0)

df=[]
for i in range(nin):
   df.append(i)
   j=i+2
   df[i]=readcsv(sys.argv[j])

### Create an Optimizer ###
def print_fun(x, f, accepted):
    print("at minima %.4f accepted %d" % (f, int(accepted)))

### Create function to compute AUC after averagings ###
def dfauc(x):
   y=0
   length=len(df)
   for i in range(len(df)):
     if i == 0:
       average=np.multiply(x[i],df[i].target.values)
       y=y+x[i]
     elif i < (len(df)-1):
       average=average+np.multiply(x[i],df[i].target.values)
       y=y+x[i]
     else:
       average=average+np.multiply(1-y,df[i].target.values)
   auc_score=ras(df[i].ref.values,average)
   return -1*auc_score 

### declare initial guess of x as x0 ###
if nin == 2:
  x0=[0.5]
elif nin ==3:
  x0=[0.3,0.3]
elif nin ==4:
  x0=[0.25,0.25,0.25]
elif nin ==5:
  x0=[0.2,0.2,0.2,0.2]
elif nin ==6:
  x0=[0.15,0.15,0.15,0.15,0.15]
elif nin ==7:
  x0=[0.1,0.1,0.1,0.1,0.1,0.1]
elif nin ==8:
  x0=[0.1,0.1,0.1,0.1,0.1,0.1,0.1]
elif nin ==9:
  x0=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]
elif nin ==10:
  x0=[0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1,0.1]

#### Module: Optimize ####
### With constraint optimize ###
if nin == 2:
  cons=({'type':'ineq','fun':lambda x: x[0]},
       {'type':'ineq','fun':lambda x: 1-x[0]})
elif nin==3:
  cons=({'type':'ineq','fun':lambda x: x[0]},
       {'type':'ineq','fun':lambda x: x[1]},
       {'type':'ineq','fun':lambda x: 1-x[0]-x[1]}
       )
elif nin==4:
  cons=({'type':'ineq','fun':lambda x: x[0]},
       {'type':'ineq','fun':lambda x: x[1]},
       {'type':'ineq','fun':lambda x: x[2]},
       {'type':'ineq','fun':lambda x: 1-x[0]-x[1]-x[2]}
       )
ret = opt.minimize(dfauc, x0, method='COBYLA',constraints=cons,tol=1e-6)

#### Best Solution ####
print("Maximum of auc is at : x = %s, its value is: f(x0) = %.4f" % (ret.x, -1*ret.fun))
