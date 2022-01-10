# Load packages:
import os
import pickle
import numpy as np
import netCDF4 as nc
from itertools import compress
import datetime as dt
from joblib import Parallel

#-------------------------------------------------------------
# Components to specify:
years=[2016,2017,2018,2019]                  # year to calculate
base = 'ANHA12_continental'                  # part of folder name structure
back = '_20211130'                           # part of folder name structure
results_folder = 'river-continental-202112/' # where to place the results

#-------------------------------------------------------------
# Set sub-domain region coordinates:
imin, imax = 1479, 2179 # because of Python zero indexing
jmin, jmax = 159, 799

# Load coordinate and mesh mask files:
mask  = nc.Dataset('/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mask.nc') 
tmask = np.array(mask.variables['tmask'])[0,:,imin:imax,jmin:jmax] 
umask = np.array(mask.variables['umask'])[0,:,imin:imax,jmin:jmax]
vmask = np.array(mask.variables['vmask'])[0,:,imin:imax,jmin:jmax]

mesh = nc.Dataset('/ocean/brogalla/GEOTRACES/data/ANHA12/ANHA12_mesh1.nc')
e1t_base  = np.array(mesh.variables['e1t'])[0,imin:imax,jmin:jmax]
e1v_base  = np.array(mesh.variables['e1v'])[0,imin:imax,jmin:jmax]
e2t_base  = np.array(mesh.variables['e2t'])[0,imin:imax,jmin:jmax]
e2u_base  = np.array(mesh.variables['e2u'])[0,imin:imax,jmin:jmax]
e3t  = np.array(mesh.variables['e3t_0'])[0,:,:,:]

e1t = np.empty_like(e3t[:,imin:imax,jmin:jmax]); e1t[:] = e1t_base
e1v = np.empty_like(e3t[:,imin:imax,jmin:jmax]); e1v[:] = e1v_base
e2t = np.empty_like(e3t[:,imin:imax,jmin:jmax]); e2t[:] = e2t_base
e2u = np.empty_like(e3t[:,imin:imax,jmin:jmax]); e2u[:] = e2u_base

e3v = np.zeros_like(e3t)
e3u = np.zeros_like(e3t)
for layer in range(50):
    for i in range(imin-5, imax+5): 
        for j in range(jmin-5, jmax+5): 
            e3v[layer, i, j] = min(e3t[layer, i, j], e3t[layer, i+1, j]) ## double-check
            e3u[layer, i, j] = min(e3t[layer, i, j], e3t[layer, i, j+1]) ## double-check

e3v = e3v[:,imin:imax,jmin:jmax]
e3u = e3u[:,imin:imax,jmin:jmax]

# coordinates of boundaries for which to calculate fluxes:----
xmin = 1480 # based on fortran coordinates which start wind index 1
ymin = 160
l1i = 2013-xmin; l1j = np.arange(300-ymin,392-ymin)
l2i = 1935-xmin; l2j = np.arange(450-ymin,530-ymin) # Western end of Parry Channel
l3i = np.arange(1850-xmin,1885-xmin); l3j = 555-ymin
l4i = np.arange(1753-xmin,1837-xmin); l4j = 568-ymin
l5i = np.arange(1720-xmin,1790-xmin); l5j = 605-ymin
l6i = 1730-xmin; l6j = np.arange(660-ymin,690-ymin)

t1i = np.arange(1635-xmin,1653-xmin); t1j = 760-ymin

r1i = 1520-xmin; r1j = np.arange(505-ymin,673-ymin)
r2i = 1520-xmin; r2j = np.arange(385-ymin,405-ymin)

N1i = np.arange(1570-xmin,1630-xmin); N1j = 635-ymin #Nares Strait
P1i = 1585-xmin; P1j = np.arange(485-ymin,538-ymin)  # Eastern end of Parry channel

#-------------------------------------------------------------
def files_time_series(base, back, year, start_date, end_date):
    #start_date and end_date are datetime objects
    
    # Create list of filenames that fall within the start and end date time range:
    dmn_file_list = np.sort(os.listdir(f'/data/brogalla/run_storage/{results_folder}{base}-{year}{back}/'))
    dyn_file_list = np.sort(os.listdir('/data/brogalla/ANHA12/'))
    
    Vlist = [i[26:31]=='gridV' for i in dyn_file_list]
    Ulist = [i[26:31]=='gridU' for i in dyn_file_list]
    Plist = [i[35:39]=='ptrc'  for i in dmn_file_list]
    
    gridV_list = list(compress(dyn_file_list, Vlist))
    gridU_list = list(compress(dyn_file_list, Ulist))
    gridP_list = list(compress(dmn_file_list, Plist))
    
    dateV_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridV_list]
    dateU_list = [dt.datetime.strptime(i[14:25], "y%Ym%md%d") for i in gridU_list]
    dateP_list = [dt.datetime.strptime(i[42:50], "%Y%m%d")    for i in gridP_list]
    
    gridV_file_list = list(compress(gridV_list, [V > start_date and V < end_date for V in dateV_list]))
    gridU_file_list = list(compress(gridU_list, [U > start_date and U < end_date for U in dateU_list]))
    gridP_file_list = list(compress(gridP_list, [P > start_date and P < end_date for P in dateP_list]))
    
    if len(gridP_file_list ) > len(gridU_file_list):
        gridP_file_list = gridP_file_list[0:-1]
        print(f'List of tracer files is longer than list of dynamics files: {len(gridP_file_list)} > {len(gridU_file_list)}')
    elif len(gridU_file_list) > len(gridP_file_list):
        diff = len(gridP_file_list) - len(gridU_file_list)
        gridU_file_list = gridU_file_list[0:diff]
        gridV_file_list = gridV_file_list[0:diff]
        print(f'List of tracer files is shorter than list of dynamics files: {len(gridP_file_list)} < {len(gridU_file_list)}')
       
    return gridV_file_list, gridU_file_list, gridP_file_list

def main_calc(base, back, year, filenameU, filenameV, filenameP): 
    # Load 5-day ptrc file
    dmn_folder = f'/data/brogalla/run_storage/{results_folder}{base}-{year}{back}/'
    dmn_file   = nc.Dataset(dmn_folder+filenameP)
    dmn        = np.array(dmn_file.variables['dissolmn'])[0,:,:,:] 
    
    # Load 5-day velocity file
    dyn_folder = '/data/brogalla/ANHA12/'
    file_u  = nc.Dataset(dyn_folder + filenameU)
    file_v  = nc.Dataset(dyn_folder + filenameV)
    u_vel   = np.array(file_u.variables['vozocrtx'])[0,:,imin:imax,jmin:jmax] 
    v_vel   = np.array(file_v.variables['vomecrty'])[0,:,imin:imax,jmin:jmax] 
    
    # For each of the boundaries, call function to calculate the flux:
    flx_mnl1, flx_Vl1 = calc_flux(l1i, l1j, dmn, u_vel, v_vel)
    flx_mnl2, flx_Vl2 = calc_flux(l2i, l2j, dmn, u_vel, v_vel)
    flx_mnl3, flx_Vl3 = calc_flux(l3i, l3j, dmn, u_vel, v_vel)
    flx_mnl4, flx_Vl4 = calc_flux(l4i, l4j, dmn, u_vel, v_vel)
    flx_mnl5, flx_Vl5 = calc_flux(l5i, l5j, dmn, u_vel, v_vel)
    flx_mnl6, flx_Vl6 = calc_flux(l6i, l6j, dmn, u_vel, v_vel)
    flx_mnt1, flx_Vt1 = calc_flux(t1i, t1j, dmn, u_vel, v_vel)
    flx_mnr1, flx_Vr1 = calc_flux(r1i, r1j, dmn, u_vel, v_vel)
    flx_mnr2, flx_Vr2 = calc_flux(r2i, r2j, dmn, u_vel, v_vel)
    flx_mnN1, flx_VN1 = calc_flux(N1i, N1j, dmn, u_vel, v_vel)
    flx_mnP1, flx_VP1 = calc_flux(P1i, P1j, dmn, u_vel, v_vel)
    
    return flx_mnl1, flx_mnl2, flx_mnl3, flx_mnl4, flx_mnl5, \
            flx_mnl6, flx_mnt1, flx_mnr1, flx_mnr2, flx_mnN1, flx_mnP1, flx_Vl1, \
            flx_Vl2, flx_Vl3, flx_Vl4, flx_Vl5, \
            flx_Vl6, flx_Vt1, flx_Vr1, flx_Vr2, flx_VN1, flx_VP1

def calc_flux(i, j, dmn, u_vel, v_vel): 
    i = np.array(i)
    j = np.array(j)

    # horizontal boundary
    if i.size > j.size: 
        bdy_vel   = u_vel[:, i[0]:i[-1], j]
        dmn_bdyl  =   dmn[:, i[0]:i[-1], j]
        dmn_bdyr  =   dmn[:, i[0]:i[-1], j+1]
        area      =   e2u[:, i[0]:i[-1], j] * e3u[:,i[0]:i[-1],j]
        cond_mask = (umask[:,i[0]:i[-1],j] < 0.1)
        
    # vertical boundary
    else: 
        bdy_vel   = v_vel[:, i  , j[0]:j[-1]]
        dmn_bdyl  =   dmn[:, i  , j[0]:j[-1]]
        dmn_bdyr  =   dmn[:, i+1, j[0]:j[-1]]
        area      =   e1v[:, i  , j[0]:j[-1]] * e3v[:,i,j[0]:j[-1]]
        cond_mask = (vmask[:,i,j[0]:j[-1]] < 0.1)
        
    # Point-wise multiplication with areas of each of the grid boxes:
    flx_V  = np.multiply(bdy_vel, area)
    
    # Mn flux for each grid cell on the U or V point on the boundary:
    dmn_bdy = 0.5*(dmn_bdyl + dmn_bdyr) 
    flx_mn  = np.multiply(dmn_bdy, flx_V)
    
    # Mask the depth levels that correspond to points on land:
    flx_mask_mn = np.ma.masked_where(cond_mask, flx_mn)
    flx_mask_V  = np.ma.masked_where(cond_mask, flx_V)
    
    return flx_mask_mn, flx_mask_V

def joblib_solver(main_calc, base, back, year, gridU_file, gridV_file, gridP_file):

    flx_mnl1, flx_mnl2, flx_mnl3, flx_mnl4, flx_mnl5, \
    flx_mnl6, flx_mnt1, flx_mnr1, flx_mnr2, flx_mnN1, flx_mnP1, flx_Vl1, \
    flx_Vl2, flx_Vl3, flx_Vl4, flx_Vl5, flx_Vl6, \
    flx_Vt1, flx_Vr1, flx_Vr2, flx_VN1, flx_VP1 = main_calc(base, back, year, gridU_file, gridV_file, gridP_file) 
    
    return flx_mnl1, flx_mnl2, flx_mnl3, flx_mnl4, flx_mnl5, flx_mnl6, flx_mnt1, flx_mnr1, flx_mnr2, \
            flx_mnN1, flx_mnP1, flx_Vl1, flx_Vl2, flx_Vl3, flx_Vl4, flx_Vl5, flx_Vl6, flx_Vt1, flx_Vr1, \
            flx_Vr2, flx_VN1, flx_VP1

#-------------------------------------------------------------

for year in years:
    # Create list of five-day files:
    print(year)
    start_date = dt.datetime(year,1,1)
    end_date   = dt.datetime(year,12,31)
    gridV_files, gridU_files, gridP_files = files_time_series(base, back, year, start_date, end_date)
    print(len(gridV_files), len(gridU_files), len(gridP_files))
    
    # call the function for each file that is within range of start date, end date
    a = len(gridV_files)
    time_series_mn1 = np.empty((a, 50, l1j.size-1)); time_series_mn2 = np.empty((a, 50, l2j.size-1)); 
    time_series_mn3 = np.empty((a, 50, l3i.size-1)); time_series_mn4 = np.empty((a, 50, l4i.size-1)); 
    time_series_mn5 = np.empty((a, 50, l5i.size-1)); time_series_mn6 = np.empty((a, 50, l6j.size-1)); 
    time_series_mn7 = np.empty((a, 50, t1i.size-1)); time_series_mn8 = np.empty((a, 50, r1j.size-1)); 
    time_series_mn9 = np.empty((a, 50, r2j.size-1)); time_series_mn10 = np.empty((a, 50, N1i.size-1));
    time_series_mn11 = np.empty((a, 50, P1j.size-1)); 

    time_series_V1 = np.empty_like(time_series_mn1); time_series_V2 = np.empty_like(time_series_mn2); 
    time_series_V3 = np.empty_like(time_series_mn3); time_series_V4 = np.empty_like(time_series_mn4); 
    time_series_V5 = np.empty_like(time_series_mn5); time_series_V6 = np.empty_like(time_series_mn6); 
    time_series_V7 = np.empty_like(time_series_mn7); time_series_V8 = np.empty_like(time_series_mn8); 
    time_series_V9 = np.empty_like(time_series_mn9); time_series_V10 = np.empty_like(time_series_mn10);
    time_series_V11 = np.empty_like(time_series_mn11);
    
    files=range(0,len(gridV_files))
    joblist=[]
    for file in files:
        positional_args=[main_calc, base, back, year, gridU_files[file], gridV_files[file], gridP_files[file]]
        keyword_args={}
        joblist.append((joblib_solver,positional_args,keyword_args))

    ncores=8
    with Parallel(n_jobs=ncores,backend='threading') as parallel:
        results = parallel(joblist)
        
    time_mn1,time_mn2,time_mn3,time_mn4,time_mn5, \
    time_mn6,time_mn7,time_mn8,time_mn9,time_mn10,time_mn11,time_V1,time_V2,time_V3,time_V4,time_V5, \
    time_V6,time_V7,time_V8,time_V9,time_V10,time_V11 = zip(*results)

    time_series_mn1[:,:,:]=time_mn1[:][:][:]
    time_series_mn2[:,:,:]=time_mn2[:][:][:]
    time_series_mn3[:,:,:]=time_mn3[:][:][:]
    time_series_mn4[:,:,:]=time_mn4[:][:][:]
    time_series_mn5[:,:,:]=time_mn5[:][:][:]
    time_series_mn6[:,:,:]=time_mn6[:][:][:]
    time_series_mn7[:,:,:]=time_mn7[:][:][:]
    time_series_mn8[:,:,:]=time_mn8[:][:][:]
    time_series_mn9[:,:,:]=time_mn9[:][:][:]
    time_series_mn10[:,:,:]=time_mn10[:][:][:]
    time_series_mn11[:,:,:]=time_mn11[:][:][:]

    time_series_V1[:,:,:]=time_V1[:][:][:]
    time_series_V2[:,:,:]=time_V2[:][:][:]
    time_series_V3[:,:,:]=time_V3[:][:][:]
    time_series_V4[:,:,:]=time_V4[:][:][:]
    time_series_V5[:,:,:]=time_V5[:][:][:]
    time_series_V6[:,:,:]=time_V6[:][:][:]
    time_series_V7[:,:,:]=time_V7[:][:][:]
    time_series_V8[:,:,:]=time_V8[:][:][:]
    time_series_V9[:,:,:]=time_V9[:][:][:]
    time_series_V10[:,:,:]=time_V10[:][:][:]
    time_series_V11[:,:,:]=time_V11[:][:][:]
    
    #Save time series to pickle (since it takes a long time to calculate):
    pickle.dump((time_series_V1, time_series_V2, time_series_V3, time_series_V4, time_series_V5, \
            time_series_V6, time_series_V7, time_series_V8, time_series_V9, time_series_V10, \
            time_series_V11, time_series_mn1, time_series_mn2, time_series_mn3, time_series_mn4, \
            time_series_mn5, time_series_mn6, time_series_mn7, time_series_mn8, time_series_mn9, \
            time_series_mn10,  time_series_mn11),\
            open(f'{results_folder}time-series-{year}.pickle','wb'))

