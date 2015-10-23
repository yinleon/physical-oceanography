import numpy as np, matplotlib.pyplot as plt, xray, gsw
#1
woa = xray.open_dataset('http://iridl.ldeo.columbia.edu/SOURCES/.NOAA/.NODC/.WOA09/'
                        '.Grid-1x1/.Annual/dods')

p0 = -10.1325 #surface pressure (dbars)
pr =  0
                      
S,T,Lat,Lon,dep = xray.broadcast_arrays\
(woa['salinity.s_an'][0],woa['temperature.t_an'][0],woa.lat*1,woa.lon*1,woa.depth*1)

p = -1*gsw.p_from_z(dep,Lat)
SA = gsw.SA_from_SP(S,p,Lon,Lat)
TC = gsw.CT_from_t(SA,T,p)
rho= gsw.rho_CT_exact(SA,TC,pr)


rhoArr = xray.DataArray(rho,dims=['depth', 'lat','lon'],coords=[woa.depth,woa.lat,woa.lon])
rhoArr1 = rhoArr.sel(lat=slice(30, 50),lon=slice(170.5,170.5))
rhoArr2 = rhoArr.sel(lat=slice(-72, -55),lon=slice(170.5,170.5))

plt.figure(figsize=(7,7)) 
plt.contourf(rhoArr1.lat,rhoArr1.depth,rhoArr1.squeeze(dim=None),cmap='ocean')
plt.title('Potential Density of Kuroshio Extension')
plt.xlabel('lat')
plt.ylabel('depth(m)')
plt.ylim(0,3000)
cbar = plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04)
cbar.ax.set_xlabel('rho  kg m^-3')
plt.show()

plt.figure(figsize=(7,7)) 
plt.contourf(rhoArr2.lat,rhoArr2.depth,rhoArr2.squeeze(dim=None),cmap='ocean')
plt.title('Potential Density Antartic Circumpolar Current')
plt.xlabel('lat')
plt.ylabel('depth(m)')
plt.xlim(-71,-55)
plt.ylim(0,3000)
cbar = plt.colorbar(orientation='vertical',fraction=0.046, pad=0.04)
cbar.ax.set_xlabel('rho  kg m^-3')
plt.show()

#print test[0]
"""
#2
# download SCOW data
!curl -O http://cioss.coas.oregonstate.edu/scow/data/monthly_fields/wind_stress_zonal_monthly_maps.nc
!curl -O http://cioss.coas.oregonstate.edu/scow/data/monthly_fields/wind_stress_meridional_monthly_maps.nc

# load data
scow_zonal = xray.open_dataset('wind_stress_zonal_monthly_maps.nc')
scow_merid = xray.open_dataset('wind_stress_meridional_monthly_maps.nc')
# each month is encoded as a different variable, annoying!
# have to manually average in time
dvars = scow_zonal.data_vars.keys()
taux = scow_zonal[dvars[0]]
tauy = scow_merid[dvars[0]]
for k in dvars[1:]:
    taux += scow_zonal[k]
    tauy = scow_merid[dvars[0]]
missing_value = -9999
taux = (taux / 12).where(taux>missing_value)
tauy = (tauy / 12).where(tauy>missing_value)

taux = taux.loc[taux.latitude<65]
taux = taux.loc[taux.latitude>-65]

rh0 = 1025           # density of seawater
om = (24*pi)/86400   # frequency
lat1 = len(taux['latitude'])-1
lon1 = len(taux['longitude'])-1
for i in range(lon1):
    for j in range(lat1):
        f = 2*om*math.sin(j)
        Vek = -(1/rh0*f)*(taux[j][i])
        Uek =  (1/rh0*f)*(tauY[j][i])
"""
#3
