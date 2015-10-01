"""
Created on Thu Sep 17 21:46:44 2015

@author: leonyin
"""

import numpy, matplotlib as m, gsw as sw,xray

SF = 2
"""****************************************************************************
1) Cabbeling
****************************************************************************"""
# You make two measurements of seawater with a CTD...
# T (in-situ temperature, C)
T1  =  0.0
T2  =  16.45
# Sp (Practical Salinity, PSU)
S1  =  31.0
S2  =  32.0

p0  =  0         # dbar pressure at surface
lat =  45        # N
lon = -30        # E

# First convert the measurments to absolute salinity 
Sa1 = sw.SA_from_SP(S1,p0,lon,lat)
Sa2 = sw.SA_from_SP(S2,p0,lon,lat)

# ...and conservative temperature.
Tc1 = sw.CT_from_t(Sa1,T1,p0)
Tc2 = sw.CT_from_t(Sa2,T2,p0)

# Now calculate the density of each water parcel? 
rho1 = sw.rho(Sa1,T1,p0)
rho2 = sw.rho(Sa2,T2,p0)

#Which water mass is denser?
print"The measurement 1 is", round(rho1-rho2,SF),"kg/m^2 denser than measurement 2."
#What is their average density?
print"Their average density is",round((rho1+rho2)/2,SF),"."

# Now allow the two water masses to mix. When they mix, they homogenize their conservative temperature and absolute salinity. 
T3  = (T1+T2)/2
Sa3 = (Sa1+Sa2)/2
# What is the density of the new water mass?
rho3 = sw.rho(Sa3,T3,p0) #sw.rho_CT() doesn't work, so I used rho(Sa,t,p)
print"The density of the new water mass is",round(rho3,SF),"."
print"The density of the new water mass is > rho1("+str(round(rho1,SF))+") and < rho2("+str(round(rho2,SF))+"), and",\
    round(rho3-(rho1+rho2)/2,SF),"kg/m^2 denser than the average of the two water masses."
    
"""****************************************************************************
2) Stratification and Thermobaricity
****************************************************************************"""
lat = -65       # S
lon = -20       # E

# T (in-situ temperature, C)
T1  = -1.8
T2  =  0.0

# Sp (Practical Salinity, PSU)
S1  = 33.0
S2  = 33.2

# P (Pressure, dbar))
p1  = 0
p2  = 20
pr  = 0 #reference pressure at the surface


Sa1=sw.SA_from_SP(S1,p1,lon,lat)
Sa2=sw.SA_from_SP(S2,p2,lon,lat)
Tc1 = sw.CT_from_t(Sa1,T1,p1)
Tc2 = sw.CT_from_t(Sa2,T2,p2)


# Assess the stability of the water column by comparing the densities of the two 
# water masses referenced to the same pressue (i.e. use potential density). 
rho1 = sw.rho_CT_exact(Sa1,Tc1,pr)
rho2 = sw.rho_CT_exact(Sa2,Tc2,pr)

# Is the water column stably stratified in this region?
print"\n\nThe water mass from the first measurement at the surface has a density of",round(rho1,SF),\
"which is",round(rho1-rho2,SF)*(-1),"less dense than water mass below ("+str(round(rho2,SF))+")...\nCreating stable stratification."

#Now imagine that ocean circulation transports the same two water masses to pressures 
#of 4990 dbar and 5010 dbar respectively. (One is still approx 20 m deeper than the other.) 
p3 =  4990
p4 =  5010
pr2 =  5000

Sa1=sw.SA_from_SP(S1,p3,lon,lat)
Sa1=sw.SA_from_SP(S2,p4,lon,lat)
Tc1 = sw.CT_from_t(Sa1,T1,p3)
Tc2 = sw.CT_from_t(Sa2,T2,p4)
#Compare the two potential densities using the mid-point reference pressure of 5000 dbar. 
#How does the stratification differ?
rho3 = sw.rho_CT_exact(Sa1,Tc1,pr2)
rho4 = sw.rho_CT_exact(Sa2,Tc2,pr2)
print"\nWhen the same two water masses are brought down ~5000 m, the upper water mass has a desnity of",round(rho3,SF),\
"which is",round(rho3-rho4,SF),"kg/m^2 more desne than the water mass below("+str(round(rho4,SF))+")...\nResulting in instability and convection."

"""****************************************************************************
3) ARGO Profile Analysis
****************************************************************************"""
argo = xray.open_dataset('http://data.nodc.noaa.gov/opendap/argo/data/pacific/2015/08/nodc_R5903323_203.nc')
lon = -101.36
lat = -66.76
print"\n\n3.1What are the dimensions of the variables pres, temp, and psal?\nall three variables are 1x56\
\nWhat are the lat/lon coordinates of the profile? Where is this in the ocean?\
\n3.2The profile is located at 66.76S and 101.36W in the Southern Ocean.\
3.3What type of python object is represented by the variable argo?\
\nxray.Dataset."

from mpl_toolkits.basemap import Basemap
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))     # initiate plot w/ dimensions.
map = Basemap(projection='hammer',lon_0=180)
map.drawcoastlines()
map.fillcontinents(color='#cc9966',lake_color='#99ffff')
map.drawmapboundary(fill_color='#99ffff')
x,y = map(lon, lat)
map.scatter(x,y,5,marker='o',color='r')
plt.title('Location of ARGO floats',fontsize=12)
plt.show()

# pressure
p = argo.pres[0]
# practical salinity
psal = argo.psal[0]
# in-situ temperature
t = argo.temp[0]


print"3.4 Absolute salinity (g/lg)"
Sa=sw.SA_from_SP(psal,p,lon,lat)
print Sa
print"3.5 Conservative temperature (C)"
Tc = sw.CT_from_t(Sa,t,p)
print Tc
print"3.6 Water column height (m)"
z=sw.z_from_p(p,lat)
print z

# Then make plots of SA and Tc vs z. Include axis labels and titles. 
lblSize = 10
fig,ax = plt.subplots(ncols=2,figsize=(8,6))   
ax[0].set_xlabel('Absolute Salinity [g/kg]',size=lblSize )
ax[0].set_ylabel('Water column height [m]',size=lblSize )
#ax[0].set_ylim((-1850,0))
#ax.set_ylim(-20,1) 
ax[0].scatter(Sa,z ,marker='o',s=12,\
                   color='g',
                    alpha=.7,zorder=10)

ax[0].plot(Sa,z,'g')
ax[0].set_title(r'$S_A$',size=16)

    
ax[1].set_xlabel(r'Conservative Temperature [$^\circ$C]',size=lblSize)
ax[1].set_ylabel('Water column height [m]',size=lblSize)
#ax[1].set_ylim(-1850,0)
#ax.set_ylim(-20,1) 
ax[1].scatter(Tc,z ,marker='o',s=12,\
                   color='g',
                    alpha=.7,zorder=10)
ax[1].plot(Tc,z,'g')
ax[1].set_title(r'$\Theta$',size=16)
[a.set_ylim([-1850,0]) for a in ax]
plt.tight_layout()



# Finally, use gsw to calculate rhosurf (surface potential density) and N2 (buoyancy frequency). 
rh0 =  sw.rho_CT_exact(Sa,Tc,p[0])
N2,pMid  =  sw.Nsquared(Sa,Tc,p,lat)

fig,ax = plt.subplots(ncols=2,figsize=(8,6))  
ax[0].set_xlabel('Surface potential density[kg m$^{-3}$]',size=lblSize)
ax[0].set_ylabel('Water column height [m]',size=lblSize)
ax[0].scatter(rh0,z ,marker='o',s=12,\
                   color='b',
                    alpha=.7,zorder=10)
ax[0].plot(rh0,z,'b')
ax[0].set_title(r'$\sigma_0$',size=16)

  
ax[1].set_xlabel('Buoyancy frequency [s$^{-2}$]',size=lblSize)
ax[1].set_ylabel('Water column height [m]',size=lblSize)
#ax[3].set_ylim(-1850,0)
ax[1].set_xlim(-1e-6,4e-5) 
ax[1].scatter(N2,z[:-1] ,marker='o',s=12,\
                   color='b',
                    alpha=.7,zorder=10)
ax[1].plot(N2,z[:-1],'b')
ax[1].set_title(r'$N^{2}$',size=16)
#ax[0].set_ylim((-1850,0))

[a.set_ylim([-1850,0]) for a in ax]
plt.tight_layout()

# Based on the figures you just made, discuss this profile a bit. Is the stratification dominated by salinity or temperature? Is it typical of the global ocean?
print"The homogenous mixed-layer occurs in the first ~100 m, followed by a steep decrease in each profile at the thermocline/halocline.\nConvection is dominated by the stratified salinity (steady gradient followed by abrupt halocline), whereas temperature is homogenous until the thermocline.\
\nThis behavior are not unique to the Southern Ocean, but rather emblematic of the vertical profile of the global ocean." 

"""****************************************************************************
4) Sensible heat flux
****************************************************************************"""
# Params
u10 = 10    # m/s wind speed at 10m
u   = 0     # m/s current speed in Ocean surf
T10 = 18    # C temp at 10m
T   = 20    # C temp at Ocean surf (note units can be K since difference is taken)
Ch  = 1e-3 # sepecific heat of seawater
CpA = 1030  # J kg-1 K-1 specific heat of air
pA  = 1.3   # kg m-3 density of air

# Calcluate sensible heat w/ units: J m-2 s-1 = W m-2
Qs = pA*CpA*Ch*abs(u10-u)*(T10-T)
#note negative flux means heat goes into the ocean...?
# Params
h   = 50    # m MLD
rho0= 1027  # kg m-3 density of seawater
s2d = 86400 # conversion from seconds -> day

# Calculate the heat flux units: W/ J K-1 = K s-1
dT = (Qs/(h*CpA*rho0))*s2d

import numpy as np
# Params
T10 = 18
T   = 20
day = 0
Tp  = T-T10 # C sea-air temperature gradient
TpList = np.zeros(118)
TpList[0] = Tp # List of sea-air temperature gradient for plotting...
Tlist =np.zeros(118)
dTlist = np.zeros(118)
days= np.zeros(118)
for i in range(118):
    days[i]=i+1
while(Tp > .15):
    Qs = pA*CpA*Ch*abs(u10-u)*(T10-T) 
    dT = (Qs/(h*CpA*rho0))*s2d
    T = T+dT
    Tp = T-T10
    print "day",day,":\nthe heat flux is",round(dT,3),"degrees/day \nthe temp gradient is",round(Tp,3)
    #book-keeping...      
    TpList[day] = T-T10
    Tlist[day] = T
    dTlist[day]=dT
    day=day+1
plt.scatter(days,TpList)
#plt.plot(Tlist,TpList)
#plt.xlim(18,19.5)
#plt.ylim(0,1.5)
plt.title("Time evolution of the air-sea T diff")
plt.ylabel('change in temp (C)')
plt.xlabel('days')
plt.show()
plt.scatter(TpList,dTlist)
plt.title("In-situ ROC vs T gradient")
plt.ylabel('ROC temp (C/sec)')
plt.xlabel('in-situ vs T10 gradient (C) ')
plt.show()


# for linear regression
#par = np.polyfit(Tlist, TpList, 1, full=True)
#m=par[0][0]     # SLOPE
#b=par[0][1]     # Y-INTERCEPT
#equation=str(round(m,2))+"x "+str(round(b,1)) 

"""
assuming that wind and current velocity remain unchanged, the temperature gradient
becomes weaker day-by-day as the heat flux is dependent on the senible heat, which
is a function of the temperature gradient.

The cooling rate is positively coorelated to the temperature gradient.

lambda is equalt to (T-dT) - T10 which is tied to velocity and temp gradients between the sea and air.
"""

"""****************************************************************************
5) Evaporation and Latent Heat Flux
****************************************************************************"""
#print"5.1 What is qsat, the specific humidity of the right at the sea surface? 
#(The water temperature is 20C, and the air right at the surface is always 100% saturated.)"
import math
T = 20+273
# Calculating (saturated) specific humidity same B/C rel Hum = 100%
qsat0 = 0.98*((6.4e5/pA))*math.exp((-5107.4)/T)
print qsat0

#5.2 Assume that the air in the cold front has 50% relative humidity at 10 m above the sea surface. What is q10? (Recall that T10=18C.)
T10 = 18+273
# Calculating specific humidity
qsat10 = 0.98*((6.4e5/pA))*math.exp((-5107.4)/T10)
# Calculating specific humidity at 50% relative humidity
q10 = qsat10/.50 #unitless or g/

#print"5.3 Calculate the evaporation rate immediately after the drop in air temperature.\
# What are the units of E? \
# Rainfall is often measured in units of mm per day. Can you convert your answer to these units?"
cE = Ch # exchange coefficient for water vapor
# calculate evaporation rate 
E   = -pA*cE*abs(u10-u)*(q10-qsat0)
Emm = E*s2d # convert to mm/day
#print"5.4 Calculate the instantaneous rate of change of the mixed layer salinity \
#due to evaporation immediately after the drop in air temperature"
SA0 = 35.0  # g/kg  average salinity of the mixed layer
P   = 0     # kg / m2 s percipitation flux
R   = 0     # kg / m2 s runoff flux

# Calculate instantaneous rate of change of MLS units: 
F=(E-P-R)*(SA0/(rho0*(1-SA0)))


print"5.5 Calculate the latent heat flux corresponding with this evaporation rate."
Le  = 2.5e6 # J / kg latent heat of vaporization
# Calculate latent heat flux units: J / m^2 s = W / m^2
Qlh = -Le*E


print"5.6 Water evaporates on the ocean surface, but percipitates downards at at higher altitutes\
with lower temperatures and saturation specific humidities. This evaporation releases latent heat,\
so Le is always negative when calculating latent heat flux."

# 6 Desnity flux rho(S,T,P) ~ rho(S,T)
