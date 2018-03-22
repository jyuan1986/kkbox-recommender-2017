####################################################
###Combine several inputs into average for Kaggle###
####################################################
####################################################
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
#### python3 *.py nin in1 in2 .... coeff1 coeff2 ...####
#### if nin=2, in1 in2 coeff1
#### if nin=3, in1 in2 in3 coeff1 coeff2
#### etc.

### Read-In All Training Data && Testing Data ###
def readcsv(iostr):
  with gzip.open(iostr,'r') as f:
       str=pd.read_csv(f)
  return str

nin=int(sys.argv[1])

#print("len(argv)=",len(sys.argv))
#for i in range(len(sys.argv)-1):
#   j=i+1
#   print("read in argv[",j,"]=",sys.argv[j])

if(nin>10):
  print("maximum input files is %d" % nin)
  print("exiting")
  exit(0)

df=[]
for i in range(nin):
   df.append(i)
   j=i+2
   df[i]=readcsv(sys.argv[j])

coeff=[]
sumcoeff=0
for i in range(nin):
    coeff.append(i)
    if i<(nin-1):
      j=i+2+nin
      #print("read in argv[",j,"]=",sys.argv[j])
      coeff[i]=float(sys.argv[j])
      sumcoeff+=coeff[i]
    else:
      coeff[i]=1-sumcoeff 

print("coeff=",coeff)


### Perform Average over coeffs ###
def print_fun(x, f, accepted):
    print("at minima %.4f accepted %d" % (f, int(accepted)))

#function to compute AUC after averaging over coeff[] #
def avecoeff(x):
   for i in range(len(df)):
     if i == 0:
       average=np.multiply(x[i],df[i].target.values)
     else:
       average=average+np.multiply(x[i],df[i].target.values)
   return average 

### Write out ###
subm = pd.DataFrame()
#subm['id'] = df[0].ref.values 
subm['ref'] = df[0].ref.values 
subm['target'] = avecoeff(coeff)
subm.to_csv('average.valid.csv.gz', compression = 'gzip', index=False, float_format = '%.5f')
print('Done!')

