# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:17:03 2023

!! WARNING RUN CELL 1 FIRST AND THEN CELL BY CELL - NOT THE WHOLE FILE AT ONCE

@author: Guillaume PIROT
"""

import numpy as np
import matplotlib.pyplot as plt
import geone as gn

from lbd_functions import get_plane_coeff,gen_lense_params,rotY,rotZ
from lbd_functions import gen_grf_planar_trend,plot3sections,plot_elevation
from lbd_functions import generate_reggrid_truncated_ellipsoid
from lbd_functions import gen_concatenated_lenses_params,gen_facies_youth_mx,get_facies_prop

runfile('.\lbd_params.py')


#%%############################################################################
# ESTIMATE PLANE COEFFICIENTS FROM 3 NON-COLINEAR POINTS
###############################################################################
nx = int((xmax-xmin)/dx)
ny = int((ymax-ymin)/dy)
vx = np.linspace(xmin+dx/2,xmax-dx/2,nx,endpoint=True)
vy = np.linspace(ymin+dy/2,ymax-dy/2,ny,endpoint=True)
yy,xx = np.meshgrid(vy,vx,indexing='ij')

a,b,c,d = get_plane_coeff(pts_basement)
zz_basement = -(d+b*yy+a*xx)/c
plt.figure(dpi=300),plt.title('zz_basement')
im=plt.imshow(zz_basement)
cbar=plt.colorbar(im)
cbar.set_label('elevation [m]')
plt.show()

a,b,c,d = get_plane_coeff(pts_topography)
zz_topography = -(d+b*yy+a*xx)/c
plt.figure(dpi=300),plt.title('zz_topography')
im=plt.imshow(zz_topography)
cbar=plt.colorbar(im)
cbar.set_label('elevation [m]')
plt.show()

#%%############################################################################
# ESTIMATE PLANE COEFFICIENTS FROM 3 NON-COLINEAR POINTS
###############################################################################
vector_ini=np.reshape(np.array([1,0,0]),(3,1))
print('rotZ([1,0,0],pi/2)')
print(str(rotZ(vector_ini,np.pi/2)))
print('rotY([1,0,0],pi/2)')
print(str(rotY(vector_ini,np.pi/2)))

#%%############################################################################
# GENERATE UNCONDITIONAL GAUSSIAN RANDOM FIELDS
###############################################################################
nx = int((xmax-xmin)/dx)
ny = int((ymax-ymin)/dy)
ox = xmin+dx/2
oy = ymin+dy/2

dimension = (nx, ny)
spacing = (dx, dy)
origin = (ox, oy)

cov_model = cov_model_basement
print(cov_model)
cov_model.plot_model(vario=True, figsize=(16,8))
plt.suptitle('Covariance model')
plt.show()

nreal = 4
np.random.seed(123)
sim2Da = gn.grf.grf2D(cov_model, dimension, spacing, origin, nreal=nreal)
im2a = gn.img.Img(nx, ny, 1, dx, dy, 1., ox, oy, 0., nv=nreal, val=sim2Da)
del(sim2Da)
plt.subplots(2, 2, figsize=(12,9),dpi=300)
for i in range(4):
    plt.subplot(2, 2, i+1)
    gn.imgplot.drawImage2D(im2a, iv=i)
    plt.title('real #{}'.format(i))
plt.tight_layout()
plt.show()

#%%############################################################################
# GENERATE UNCONDITIONAL GAUSSIAN RANDOM FIELDS + ADD PLANAR TREND
###############################################################################
nx = int((xmax-xmin)/dx)
ny = int((ymax-ymin)/dy)
ox = xmin+dx/2
oy = ymin+dy/2

vx = np.linspace(xmin+dx/2,xmax-dx/2,nx,endpoint=True)
vy = np.linspace(ymin+dy/2,ymax-dy/2,ny,endpoint=True)

basement = gen_grf_planar_trend(vx,vy,pts_basement,cov_model_basement,lbdseed)
plot_elevation(basement,'basement','m',vx,vy)


#%%############################################################################
# STOCHASTIC GENERATION OF TRUNCATED ELLIPSOID PARAMETERS
###############################################################################
nx = int((xmax-xmin)/dx)
ny = int((ymax-ymin)/dy)
nz = int((zmax-zmin)/dz)
curLenseParams = sandLenses
maskLenses = np.ones((nz,ny,nx))

ix,iy,iz,dipdir,dip,a,b,c,truncr = gen_lense_params(curLenseParams,maskLenses,dx,dz,lbdseed)

plt.subplots(3,3,figsize=(15,15),dpi=300)
plt.subplot(3,3,1),plt.hist(ix),plt.title('x index')
plt.subplot(3,3,2),plt.hist(iy),plt.title('y index')
plt.subplot(3,3,3),plt.hist(iz),plt.title('z index')
plt.subplot(3,3,4),plt.hist(truncr),plt.title('truncation ratio')
plt.subplot(3,3,5),plt.hist(dipdir),plt.title('dip direction [$\circ$]')
plt.subplot(3,3,6),plt.hist(dip),plt.title('dip [$\circ$]')
plt.subplot(3,3,7),plt.hist(a),plt.title('a [m]')
plt.subplot(3,3,8),plt.hist(b),plt.title('b [m]')
plt.subplot(3,3,9),plt.hist(c),plt.title('c [m]')
plt.show()

#%%############################################################################
# TRUNCATED ELLIPSOID GENERATION
###############################################################################
nx = int((xmax-xmin)/dx)
ny = int((ymax-ymin)/dy)
nz = int((zmax-zmin)/dz)
ix,iy,iz,dipdir,dip,a,b,c,truncr = 30,22,50,90*np.pi/180,0*np.pi/180,4000,2000,10,0.2

vx = np.linspace(xmin+dx/2,xmax-dx/2,nx,endpoint=True)
vy = np.linspace(ymin+dy/2,ymax-dy/2,ny,endpoint=True)
vz = np.linspace(zmin+dz/2,zmax-dz/2,nz,endpoint=True)

mx_trunc_ellipsoid,izyx = generate_reggrid_truncated_ellipsoid(vx,vy,vz,ix,iy,iz,dipdir,dip,a,b,c,truncr)
# check plot of sections
section_zy = np.reshape(mx_trunc_ellipsoid[:,:,ix],(nz,ny))
section_zx = np.reshape(mx_trunc_ellipsoid[:,iy,:],(nz,nx))
section_yx = np.reshape(mx_trunc_ellipsoid[iz-int(np.ceil(c*truncr/dz*1.1)),:,:],(ny,nx))

plt.figure(dpi=300),plt.xlabel('y (hm)'),plt.ylabel('z (m)')
plt.imshow(section_zy,origin='lower',extent=(ymin/1E2,ymax/1E2,zmin,zmax))

plt.figure(dpi=300),plt.xlabel('x (hm)'),plt.ylabel('z (m)')
plt.imshow(section_zx,origin='lower',extent=(xmin/1E2,xmax/1E2,zmin,zmax))

plt.figure(dpi=300),plt.xlabel('x (km)'),plt.ylabel('y (km)')
plt.imshow(section_yx,origin='lower',extent=(xmin/1E3,xmax/1E3,ymin/1E3,ymax/1E3))

#%%############################################################################
# OBJECT BASE MODEL
###############################################################################

nx = int((xmax-xmin)/dx)
ny = int((ymax-ymin)/dy)
nz = int((zmax-zmin)/dz)
vx = np.linspace(xmin+dx/2,xmax-dx/2,nx,endpoint=True)
vy = np.linspace(ymin+dy/2,ymax-dy/2,ny,endpoint=True)
vz = np.linspace(zmin+dz/2,zmax-dz/2,nz,endpoint=True)
zzz,yyy,xxx = np.meshgrid(vz,vy,vx,indexing='ij')

# GENERATE BASEMENT 
basement = gen_grf_planar_trend(vx,vy,pts_basement,cov_model_basement,lbdseed+1)
plot_elevation(basement,'basement','m',vx,vy)

# GENERATE TOPOGRAPHY
topography = gen_grf_planar_trend(vx,vy,pts_topography,cov_model_topography,lbdseed+2)
plot_elevation(topography,'topography','m',vx,vy)

# CREATE AQUIFER MASK
tmp_bst = np.tile(np.reshape(basement,(1,ny,nx)),(nz,1,1))
tmp_top = np.tile(np.reshape(topography,(1,ny,nx)),(nz,1,1))
mask_aquifer = np.zeros((nz,ny,nx))
ix_aq = np.where((zzz>=tmp_bst)&(zzz<=tmp_top))
mask_aquifer[ix_aq] = 1
ix_air = np.where(zzz>tmp_top)
mask_aquifer[ix_air] = np.nan

# GENERATE TRUNCATED ELLIPSOID PARAMS FOR SAND LENSES AND CLAY LENSES
lenseClassesList = [sandLenses,clayLenses]
Lix,Liy,Liz,Ldipdir,Ldip,La,Lb,Lc,Ltruncr,Lfacies = gen_concatenated_lenses_params(lenseClassesList,mask_aquifer,dx,dz,lbdseed+3)

# GENERATE OBJECT BASED MODELS
facies_mx,relative_youth_mx = gen_facies_youth_mx(vx,vy,vz,Lix,Liy,Liz,Ldipdir,Ldip,La,Lb,Lc,Ltruncr,Lfacies,mask_aquifer)

# COMPUTE COUNT AND PROPORTIONS PER LITHOCODE
prop = get_facies_prop(facies_mx,mask_aquifer,lithocodes[1:],labels[1:])

# check plot of sections
from matplotlib.colors import ListedColormap
cmap = ListedColormap(["black", "grey", "gold", "sienna"])
ix,iy,iz=30,22,50
# plot3sections(mask_aquifer,ix,iy,iz,'aquifer mask','viridis',vx,vy,vz)
plot3sections(facies_mx,ix,iy,iz,'facies',cmap,vx,vy,vz,prop.loc[:,['proportion','labels']])
plot3sections(relative_youth_mx,ix,iy,iz,'relative youth','Greys',vx,vy,vz)

