## Author Akash Ghosh, Pine Group, NYU
## For internal Research Only

import numpy as np
import matplotlib.pyplot as plt
import math


def P_Implicit_Dep(r1,r2,sigma_g,L,phi,rdep,rcc):
    
    # sigmaA is the particle size
    # sigmaA2 is another particle size
    # sigma_g is the brush density
    # L is the brush length
    # rcc is the centre to centre distance
    
    r1_eff = L + r1
    r2_eff = L + r2
    x = rcc - r1 - r2
    q1 = r1_eff/rdep
    q2 = r2_eff/rdep
    n = rcc/rdep
    r_Avg = 2/(1/r1+1/r2)
    if ((x <= 2*L and x >0) and (rcc <=(r1_eff + r2_eff + 2*rdep))):
        U_B = 16*math.pi*L**2*r_Avg*sigma_g**(3/2)/35*(28*((2*L/x)**(1/4)-1)+(20/11)*(1-(x/(2*L))**(11/4))+12*(x/(2*L)-1))
        U_AO = -phi/16*(q1+q2+2-n)**2*(n+2*(q1+q2+2)-3/n*(q1**2+q2**2-2*q1*q2))
    elif (( x > 2*L) and (rcc <= r1_eff + r2_eff + 2*rdep)):
        U_B = 0
        U_AO = -phi/16*(q1+q2+2-n)**2*(n+2*(q1+q2+2)-3/n*(q1**2+q2**2-2*q1*q2))
    else:
        U_AO = 0
        U_B = 0
    U = U_B + U_AO
    return U

def generate_U_EAO(rp, rn, L, sigmag, sigmap, phi):
    sigmac = rp + rn
    rc = sigmac + sigmap + 2*L
    rcc = np.linspace(sigmac + 0.01, sigmac + 2*L + sigmap + 0.0015, 300)
    U_EAO = np.zeros_like(rcc, dtype=float)
    for i, r in enumerate(rcc):
        U_EAO[i] = P_Implicit_Dep(rp, rn, sigmag, L, phi, sigmap/2, r)   # use args, not globals
    return rcc, rc, U_EAO

    
rp = 0.5                 # radii -> contact at 1.30
rn = 0.605

sigmac = rp+rn                # contact center distance
L = 0.01                      # brush length
sigmag = 0.005*(1000)**2       # grafting density (units consistent with L)
sigmap = 0.02
rc = sigmac + 2*L + sigmap        
phi = 0.05                 # polymer volume fraction

rcc = np.linspace(sigmac + 0.01 , sigmac+ 2*L+ sigmap+0.0015, 300)
# U_EAO = generate_U_EAO(rp, rn, L, sigmag, sigmap, phi)
U_EAO= np.zeros_like(rcc, dtype =float)

for i,r in enumerate(rcc):
    # print(i, r)
    U_EAO[i] = P_Implicit_Dep(rp,rn, sigmag,L,phi,sigmap/2,r)

U_min = np.min(U_EAO)
plt.plot(rcc, U_EAO,label="AO+ADG, brush length = "+str(L)+" ")
plt.plot(rcc, np.zeros(len(rcc)),'-')
plt.ylim([-6,5])

# print(f"Minimum energy of the potential is {U_min}")

plt.savefig("/home/akash/hoomd-results/AO+ADG_"+str(sigmap*1e3)+"_nm_depletant.png")
# plt.show()











