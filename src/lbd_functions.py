# -*- coding: utf-8 -*-
"""
Created on Wed Feb 15 09:15:45 2023

@author: Guillaume PIROT
"""
from datetime import datetime
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import rotZ, rotY, get_plane_coeff, get_ecdf_area_difference, get_ecdf, interp_ecdf_normalize_x, interpolate_ecdf
# from matplotlib.colors import LightSource
import matplotlib as mpl
import seaborn as sns
import pickle

def printtimelog(logtext):
    print((datetime.now()).strftime('%d-%b-%Y (%H:%M:%S)')+' - '+logtext)
    return

def printtimelogwithstarline(logtext):
    print('**********************************************************************')
    printtimelog(logtext)
    print('**********************************************************************')
    return

upaq_sandlenses_proportion = 22.5
upaq_sandlenses_proportion_tol = 2.5
upaq_sandlenses_truncation = 22.5
upaq_sandlenses_truncation_tol = 7.5
upaq_sandlenses_length = 4500
upaq_sandlenses_length_tol = 3000
upaq_sandlenses_lengthdist = 'uniform'
upaq_sandlenses_anisotropy = 4
upaq_sandlenses_anisotropy_tol = 2
upaq_sandlenses_anisotropydist = 'uniform'
upaq_sandlenses_dipdir = 90
upaq_sandlenses_dipdir_tol = 10
upaq_sandlenses_dipdirdist = 'normal'
upaq_sandlenses_dip = 0
upaq_sandlenses_dip_tol = 0
upaq_sandlenses_dipdist = 'normal'

class Lense:
    def __init__(self,proportion,proportion_tol,
                 truncation,truncation_tol,truncationdist,
                 length,length_tol,lengthdist,
                 thickness,thickness_tol,thicknessdist,
                 anisotropy,anisotropy_tol,anisotropydist,
                 dipdir,dipdir_tol,dipdirdist,
                 dip,dip_tol,dipdist,facies):
        self.proportion = proportion
        self.proportion_tol = proportion_tol
        self.truncation = truncation
        self.truncation_tol = truncation_tol
        self.truncationdist = truncationdist
        self.length = length
        self.length_tol = length_tol
        self.lengthdist = lengthdist
        self.thickness = thickness
        self.thickness_tol = thickness_tol
        self.thicknessdist = thicknessdist
        self.anisotropy = anisotropy
        self.anisotropy_tol = anisotropy_tol
        self.anisotropydist = anisotropydist
        self.dipdir = dipdir
        self.dipdir_tol = dipdir_tol
        self.dipdirdist = dipdirdist
        self.dip = dip
        self.dip_tol = dip_tol
        self.dipdist = dipdist
        self.facies = facies

def gen_rnd_params(distribution,valuemean,valuetol,nparams,seed):
    rng = np.random.default_rng(seed+1)
    if distribution == 'gaussian':
        params = rng.normal(loc=valuemean, scale=valuetol/3, size=nparams) 
    else:
        params = rng.uniform(low=valuemean-valuetol, high=valuemean+valuetol, size=nparams)     
    return params

def gen_lense_params(lenseParams,mask,dx,dz,seed):
    rng = np.random.default_rng(seed)
    nmaxLenses = np.round( lenseParams.proportion * np.sum(mask) / 
                      ((lenseParams.length-lenseParams.length_tol/2)/dx)**2 * lenseParams.anisotropy /
                      (lenseParams.thickness/dz) ).astype(int)
    (nz,ny,nx) = mask.shape
    ix = rng.integers(low=0, high=nx, size=nmaxLenses, dtype=int, endpoint=False)
    iy = rng.integers(low=0, high=ny, size=nmaxLenses, dtype=int, endpoint=False)
    iz = rng.integers(low=0, high=nz, size=nmaxLenses, dtype=int, endpoint=False)        
    xtrunc  = gen_rnd_params(lenseParams.lengthdist,     lenseParams.length,      lenseParams.length_tol,      nmaxLenses,seed+1)
    ztrunc  = gen_rnd_params(lenseParams.thicknessdist,  lenseParams.thickness,   lenseParams.thickness_tol,   nmaxLenses,seed+2)
    aniso   = gen_rnd_params(lenseParams.anisotropydist, lenseParams.anisotropy,  lenseParams.anisotropy_tol,  nmaxLenses,seed+3)
    truncr  = gen_rnd_params(lenseParams.truncationdist, lenseParams.truncation,  lenseParams.truncation_tol,  nmaxLenses,seed+4)
    dipdir  = gen_rnd_params(lenseParams.dipdirdist,     lenseParams.dipdir,      lenseParams.dipdir_tol,      nmaxLenses,seed+5)
    dip     = gen_rnd_params(lenseParams.dipdist,        lenseParams.dip,         lenseParams.dip_tol,         nmaxLenses,seed+6)
    theta = np.arccos(1-truncr)
    a = xtrunc/np.sin(theta)
    b = a/aniso
    c = ztrunc/(1-truncr)   
    return ix,iy,iz,dipdir,dip,a,b,c,truncr   


def generate_reggrid_truncated_ellipsoid(vx,vy,vz,ix,iy,iz,dipdir,dip,a,b,c,truncr):
    nx = len(vx)
    ny = len(vy)
    nz = len(vz)
    # Center of ellipsoid
    elOx = vx[ix]
    elOy = vy[iy]
    elOz = vz[iz]    
    # Translation
    vxT = vx-elOx
    vyT = vy-elOy
    vzT = vz-elOz    
    # get a first limited subset (parallelogram)
    tmpix = np.asarray(np.where(np.abs(vxT)<=a)).flatten()
    tmpiy = np.asarray(np.where(np.abs(vyT)<=a)).flatten()
    tmpiz = np.asarray(np.where(np.abs(vzT)<=(a*np.tan(dip))+c*np.cos(dip))).flatten()
    vxTss = vxT[tmpix]
    vyTss = vyT[tmpiy]
    vzTss = vzT[tmpiz]    
    # subset mesh of coordinates
    zzTss,yyTss,xxTss = np.meshgrid(vzTss,vyTss,vxTss, indexing='ij')
    # subset mesh of grid index
    izz_ss,iyy_ss,ixx_ss = np.mgrid[tmpiz[0]:tmpiz[-1]+1,tmpiy[0]:tmpiy[-1]+1,tmpix[0]:tmpix[-1]+1]
    # subset points and coordinates
    xyzTss = np.vstack((xxTss.flatten(),yyTss.flatten(),zzTss.flatten()))        
    # -(np.pi/2-dipdir) Rotation followed by -dip rotation
    xyzTssR2 = rotY(rotZ(xyzTss,-(np.pi/2-dipdir)),-dip)    
    # GET TRUNCATING PLANE
    theta = np.arccos(-truncr) # angle coordinate of plane truncating ellipsoid
    # generate three points on the surface of the ellipsoid at the desired ztrunc with phi in 0, pi/2 pi
    threepts = np.array([[a*np.sin(theta)*np.cos(0),   a*np.sin(theta)*np.cos(np.pi/2),    a*np.sin(theta)*np.cos(np.pi)],
                         [a*np.sin(theta)*np.sin(0),   a*np.sin(theta)*np.sin(np.pi/2),    a*np.sin(theta)*np.sin(np.pi)],
                         [c*np.cos(theta),             c*np.cos(theta),                    c*np.cos(theta)]])
    # -dip rorations
    threeptsR = rotY(threepts,-dip)
    p_a,p_b,p_c,p_d = get_plane_coeff(threeptsR)    
    # identify pixelid within truncated ellipsoid 
    eq_ellipse = (xyzTssR2[0,:]/a)**2 + (xyzTssR2[1,:]/b)**2 + (xyzTssR2[2,:]/c)**2 <=1
    eq_plane = (p_a*xyzTssR2[0,:] + p_b*xyzTssR2[1,:] +p_c*xyzTssR2[2,:]+p_d)<=0
    ix_trunc_ellipse = np.where(np.reshape(eq_ellipse & eq_plane,xxTss.shape))
    izz=izz_ss[ix_trunc_ellipse]
    iyy=iyy_ss[ix_trunc_ellipse]
    ixx=ixx_ss[ix_trunc_ellipse]
    izyx = tuple(np.vstack((izz,iyy,ixx)))
    mx_trunc_ellipsoid = np.zeros((nz,ny,nx))
    mx_trunc_ellipsoid[izyx]=1
    return mx_trunc_ellipsoid,izyx


def plot_elevation(dtm,title,unit,vx,vy):
    dx=vx[1]-vx[0]
    dy=vy[1]-vy[0]
    plt.figure(dpi=300)
    plt.title(title)
    im=plt.imshow(dtm,cmap='terrain',origin='lower',extent=[(vx[0]-dx/2)/1E3,(vx[-1]+dx/2)/1E3,(vy[0]-dy/2)/1E3,(vy[-1]+dy/2)/1E3])
    plt.xlabel('x(km)')
    plt.ylabel('y(km)')
    cbar=plt.colorbar(im)
    cbar.set_label('elevation ['+unit+']')
    plt.show()
    return

def plot3sections(mx,ix,iy,iz,cmap,vx,vy,vz,labeltype='facies',figsize=None,fontsize=None,figFileName=None):
    if figsize is None: figsize = (30/2.54,24/2.54)
    if fontsize is None: fontsize = 10
    if labeltype=='age':
        cmap = mpl.colormaps['Greys']

    val = np.unique(mx)
    valnotnan = val[np.where(np.isnan(val)==False)]
    
    scilim_x = len(str(int(np.maximum(np.abs(vx[0]),np.abs(vx[-1])))))-1
    scilim_y = len(str(int(np.maximum(np.abs(vy[0]),np.abs(vy[-1])))))-1
    
    nz,ny,nx = mx.shape
    dx=vx[1]-vx[0]
    dy=vy[1]-vy[0]
    dz=vz[1]-vz[0]
    xmin=vx[0]-dx/2
    xmax=vx[-1]+dx/2
    ymin=vy[0]-dy/2
    ymax=vy[-1]+dy/2
    zmin=vz[0]-dz/2
    zmax=vz[-1]+dz/2
    vmin = mx[~np.isnan(mx)].min()-0.5
    vmax = mx[~np.isnan(mx)].max()+0.5
    if labeltype=='facies':
        vmin = -1.5
        vmax = 3.5
        facies_seq = [-1,0,1,2,3]
    section_zy = np.reshape(mx[:,:,ix],(nz,ny))
    section_zx = np.reshape(mx[:,iy,:],(nz,nx))
    section_yx = np.reshape(mx[iz,:,:],(ny,nx))
    
    plt.rc("axes", linewidth=0.5) # so lines on edges of plots aren't too thick
    plt.matplotlib.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'}) # so that Arial is used
    plt.rcParams.update({'font.size': fontsize})    # size 10 font
 
    # fig, ax = plt.subplots(5, 2, figsize=(15,9),dpi=300) #fig, ax = 
    fig = plt.figure(figsize=figsize,dpi=300, layout='compressed') # figsize=(15/2.54,12/2.54), layout='compressed'
    
    #plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    #plt.rcParams['axes.titlepad'] = 5  # pad is in points...
    
   # gs = fig.add_gridspec(35,2)
    # ax1 = fig.add_subplot(gs[0:14, 0])
    # ax2 = fig.add_subplot(gs[0:14, 1])
    # ax3 = fig.add_subplot(gs[17:31, 0])
    # ax4 = fig.add_subplot(gs[17:31, 1],projection='3d',)
    # ax5 = fig.add_subplot(gs[34, :])
    
    gs = fig.add_gridspec(4,2, height_ratios=[15,15,1,1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax4 = fig.add_subplot(gs[1, 1],projection='3d',)
    ax5 = fig.add_subplot(gs[3, :])
    
    # ax2.sharey(ax1)
    # ax3.sharex(ax1)
    
    ax1.tick_params(which='both',direction='inout', width=0.5)
    ax1.set_xlabel('x - Easting [m]'),ax1.set_ylabel('z - Elevation [m]'),ax1.set_title('(a) vertical section at y='+'%.4g' % vy[iy])
    ax1.ticklabel_format(style='sci', axis='x', scilimits=(scilim_x,scilim_x))
    im=ax1.imshow(section_zx,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,extent=(xmin,xmax,zmin,zmax),aspect=100)
    
    ax2.tick_params(which='both',direction='inout', width=0.5)
    ax2.set_xlabel('y - Northing [m]'),ax2.set_ylabel('z - Elevation [m]'),ax2.set_title('(b) vertical section at x='+'%.4g' % vx[ix])
    ax2.ticklabel_format(style='sci', axis='x', scilimits=(scilim_y,scilim_y))
    ax2.imshow(section_zy,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,extent=(ymin,ymax,zmin,zmax),aspect=100)
    
    ax3.tick_params(which='both',direction='inout', width=0.5)
    ax3.set_xlabel('x - Easting [m]'),ax3.set_ylabel('y - Northing [m]'),ax3.set_title('(c) horizontal section at z='+str(vz[iz])+'m')
    ax3.ticklabel_format(style='sci', axis='x', scilimits=(scilim_x,scilim_x))
    ax3.ticklabel_format(style='sci', axis='y', scilimits=(scilim_y,scilim_y))
    ax3.imshow(section_yx,origin='lower',cmap=cmap,vmin=vmin,vmax=vmax,extent=(xmin,xmax,ymin,ymax))
    
    rgba_zx = np.array([37,41,88,255])/255
    rgba_zy = np.array([43,44,170,255])/255
    rgba_xy = np.array([72,113,247,255])/255
    
    # change all spines
    for axis in ['top','bottom','left','right']:
        ax1.spines[axis].set_linewidth(2)
        ax1.spines[axis].set_color(rgba_zx)
        ax2.spines[axis].set_linewidth(2)
        ax2.spines[axis].set_color(rgba_zy)
        ax3.spines[axis].set_linewidth(2)
        ax3.spines[axis].set_color(rgba_xy)
    
    # # PLOT SECTION LINES - !!!!! DOES NOT WORK PROPERLY: OVERPRINTED BY THE SURFACE PLOTS !!!!! 
    # x_sec_zx = np.array([vx[0]-2*dx,vx[-1]+2*dx,vx[-1]+2*dx])
    # y_sec_zx = vy[iy] * np.ones(3)
    # z_sec_zx = np.array([vz[-1]+2*dz,vz[-1]+2*dz,vz[0]-2*dz])
    # ax4.plot(x_sec_zx, y_sec_zx, z_sec_zx, '-og',linewidth = 2, alpha=0.5)
    
    # # PLOT SECTION SECTIONS  - !!!!! DOES NOT WORK PROPERLY: OVERPRINTED BY THE OTHER SURFACE PLOTS NO MATTER IF PLOTTING IT BEFORE OR AFTER !!!!! 
    # slice_vx = np.ones(len(vx)+2)*np.nan
    # slice_vy = np.ones(len(vy)+2)*np.nan
    # slice_vz = np.ones(len(vz)+2)*np.nan
    # slice_vx[1:-1] = vx
    # slice_vy[1:-1] = vy
    # slice_vz[1:-1] = vz
    # slice_vx[0] = vx[0]-dx
    # slice_vx[-1] = vx[-1]+dx
    # slice_vy[0] = vy[0]-dy
    # slice_vy[-1] = vy[-1]+dy
    # slice_vz[0] = vz[0]-dz
    # slice_vz[-1] = vz[-1]+dz    
    
    # yy_sec_zy,zz_sec_zy = np.meshgrid(slice_vy,slice_vz)
    # xx_sec_zy = np.ones(yy_sec_zy.shape) * vx[ix]
    # surf = ax4.plot_surface(xx_sec_zy,yy_sec_zy,zz_sec_zy, color=rgba_zy) #, cmap=cmap
    
    
    for j in range(25):
        iztmp = -j
        section_yx = np.reshape(mx[iztmp,:,:],(ny,nx))
        xx_sec_yx,yy_sec_yx = np.meshgrid(vx,vy)
        zz_sec_yx = np.ones(xx_sec_yx.shape) * vz[iztmp]
        section_yx_clr = np.ones((ny*nx,4))*np.nan
        for i in range(len(valnotnan)):
            if labeltype=='facies':
                section_yx_clr[np.where(section_yx.flatten()==valnotnan[i]),:] = cmap(np.where(facies_seq==valnotnan[i]))
            if labeltype=='age':
                section_yx_clr[np.where(section_yx.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
        section_yx_clr[np.where((xx_sec_yx.flatten()==vx[ix])&(np.isnan(section_yx.flatten())==False)),:]=np.array(rgba_zy) # zy sec
        section_yx_clr[np.where((yy_sec_yx.flatten()==vy[iy])&(np.isnan(section_yx.flatten())==False)),:]=np.array(rgba_zx) # zx sec
        section_yx_clr = np.reshape(section_yx_clr,(ny,nx,4))
        surf = ax4.plot_surface(xx_sec_yx,yy_sec_yx,zz_sec_yx, facecolors=section_yx_clr) #, cmap=cmap
    
    ixtmp = 0
    section_zy = np.reshape(mx[:,:,ixtmp],(nz,ny))
    yy_sec_zy,zz_sec_zy = np.meshgrid(vy,vz)
    xx_sec_zy = np.ones(yy_sec_zy.shape) * vx[ixtmp]
    section_zy_clr = np.ones((nz*ny,4))*np.nan
    for i in range(len(valnotnan)):
        if labeltype=='facies':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(np.where(facies_seq==valnotnan[i]))
        if labeltype=='age':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
            
    section_zy_clr[np.where((zz_sec_zy.flatten()==vz[iz])&(np.isnan(section_zy.flatten())==False)),:]=np.array(rgba_xy) # xy sec
    section_zy_clr[np.where((yy_sec_zy.flatten()==vy[iy])&(np.isnan(section_zy.flatten())==False)),:]=np.array(rgba_zx) # zx sec
    section_zy_clr = np.reshape(section_zy_clr,(nz,ny,4))
    surf = ax4.plot_surface(xx_sec_zy,yy_sec_zy,zz_sec_zy, facecolors=section_zy_clr) #, cmap=cmap
    
    iytmp = 0
    section_zx = np.reshape(mx[:,iytmp,:],(nz,nx))
    xx_sec_zx,zz_sec_zx = np.meshgrid(vx,vz)
    yy_sec_zx = np.ones(xx_sec_zx.shape) * vy[iytmp]
    section_zx_clr = np.ones((nz*nx,4))*np.nan
    for i in range(len(valnotnan)):
        if labeltype=='facies':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(np.where(facies_seq==valnotnan[i]))
        if labeltype=='age':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
    
    section_zx_clr[np.where((zz_sec_zx.flatten()==vz[iz])&(np.isnan(section_zx.flatten())==False)),:]=np.array(rgba_xy) # xy sec
    section_zx_clr[np.where((xx_sec_zx.flatten()==vx[ix])&(np.isnan(section_zx.flatten())==False)),:]=np.array(rgba_zy) # zy sec
    section_zx_clr = np.reshape(section_zx_clr,(nz,nx,4))
    surf = ax4.plot_surface(xx_sec_zx,yy_sec_zx,zz_sec_zx, facecolors=section_zx_clr) #, cmap=cmap
    
    ixtmp = -1
    section_zy = np.reshape(mx[:,:,ixtmp],(nz,ny))
    yy_sec_zy,zz_sec_zy = np.meshgrid(vy,vz)
    xx_sec_zy = np.ones(yy_sec_zy.shape) * vx[ixtmp]
    section_zy_clr = np.ones((nz*ny,4))*np.nan
    for i in range(len(valnotnan)):
        #section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i)
        if labeltype=='facies':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(np.where(facies_seq==valnotnan[i]))
        if labeltype=='age':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
    
    section_zy_clr[np.where((zz_sec_zy.flatten()==vz[iz])&(np.isnan(section_zy.flatten())==False)),:]=np.array(rgba_xy) # xy sec
    section_zy_clr[np.where((yy_sec_zy.flatten()==vy[iy])&(np.isnan(section_zy.flatten())==False)),:]=np.array(rgba_zx) # zx sec
    section_zy_clr = np.reshape(section_zy_clr,(nz,ny,4))
    surf = ax4.plot_surface(xx_sec_zy,yy_sec_zy,zz_sec_zy, facecolors=section_zy_clr) #, cmap=cmap
    # ls = LightSource(azdeg=150)
    # illuminated_surface = ls.shade_rgb(section_zy_clr[:,:,:-1], zz_sec_zy)
    # surf = ax4.plot_surface(xx_sec_zy,yy_sec_zy,zz_sec_zy, facecolors=illuminated_surface) #, cmap=cmap
    
    iytmp = -1
    section_zx = np.reshape(mx[:,iytmp,:],(nz,nx))
    xx_sec_zx,zz_sec_zx = np.meshgrid(vx,vz)
    yy_sec_zx = np.ones(xx_sec_zx.shape) * vy[iytmp]
    section_zx_clr = np.ones((nz*nx,4))*np.nan
    for i in range(len(valnotnan)):
        if labeltype=='facies':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(np.where(facies_seq==valnotnan[i]))
        if labeltype=='age':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
    
    section_zx_clr[np.where((zz_sec_zx.flatten()==vz[iz])&(np.isnan(section_zx.flatten())==False)),:]=np.array(rgba_xy) # xy sec
    section_zx_clr[np.where((xx_sec_zx.flatten()==vx[ix])&(np.isnan(section_zx.flatten())==False)),:]=np.array(rgba_zy) # zy sec
    section_zx_clr = np.reshape(section_zx_clr,(nz,nx,4))
    surf = ax4.plot_surface(xx_sec_zx,yy_sec_zx,zz_sec_zx, facecolors=section_zx_clr) #, cmap=cmap
    
    ax4.ticklabel_format(style='sci', axis='x', scilimits=(scilim_x,scilim_x))
    ax4.ticklabel_format(style='sci', axis='y', scilimits=(scilim_y,scilim_y))
    ax4.set_xticks([])
    ax4.set_yticks([])
    ax4.set_zticks([])
    ax4.set_xlabel('Easting [m]'),ax4.set_ylabel('Northing [m]'),ax4.set_zlabel('z')
    ax4.set_aspect('equalxy')
    ax4.view_init(25, -50, 0)
    ax4.grid(False)
    #plt.rcParams['axes.titley'] = 1.0    # y is in axes-relative coordinates.
    #plt.rcParams['axes.titlepad'] = -5  # pad is in points...
    if labeltype=='facies': ax4.set_title('(d) 3D stochastic facies model',pad=-5)
    if labeltype=='age': ax4.set_title('(d) 3D relative age model',pad=-5)
    
    fig.colorbar(im, cax=ax5,shrink=0.6,orientation='horizontal') #,shrink=0.6,fraction=.5
    if labeltype=='facies':
        
        ax5.set_xticks(facies_seq)
        ax5.set_xticklabels(['basement', 'weathered-b', 'intermediate','coarse-grained','fine-grained'])  # vertically oriented colorbar  rotation = 45
    if labeltype=='age':
        ax5.set_xlabel('relative age')
    
    if figFileName is not None:
        plt.savefig(figFileName,bbox_inches='tight')
    plt.show()
    return 

def plot3Dreal(mx,cmap,vx,vy,vz,labeltype='facies',axisOff=False,figsize=None,fontsize=None,figFileName=None,):
    if figsize is None: figsize = (30/2.54,24/2.54)
    if fontsize is None: fontsize = 10
    if labeltype=='age':
        cmap = mpl.colormaps['Greys']
    
    val = np.unique(mx)
    valnotnan = val[np.where(np.isnan(val)==False)]
    
    scilim_x = len(str(int(np.maximum(np.abs(vx[0]),np.abs(vx[-1])))))-1
    scilim_y = len(str(int(np.maximum(np.abs(vy[0]),np.abs(vy[-1])))))-1
    
    nz,ny,nx = mx.shape
    dx=vx[1]-vx[0]
    dy=vy[1]-vy[0]
    dz=vz[1]-vz[0]
    xmin=vx[0]-dx/2
    xmax=vx[-1]+dx/2
    ymin=vy[0]-dy/2
    ymax=vy[-1]+dy/2
    zmin=vz[0]-dz/2
    zmax=vz[-1]+dz/2
    vmin = mx[~np.isnan(mx)].min()-0.5
    vmax = mx[~np.isnan(mx)].max()+0.5
    
    plt.rc("axes", linewidth=0.5) # so lines on edges of plots aren't too thick
    plt.matplotlib.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'}) # so that Arial is used
    plt.rcParams.update({'font.size': fontsize})    # size 10 font
    
    # im = plt.imshow(mx[:,:,0],cmap=cmap,origin='lower')
    
    # fig, ax = plt.subplots(5, 2, figsize=(15,9),dpi=300) #fig, ax = 
    fig = plt.figure(figsize=figsize,dpi=300, layout='compressed') # figsize=(15/2.54,12/2.54), layout='compressed'  
    # gs = fig.add_gridspec(1,1, height_ratios=[15,15,1,1])
    ax = fig.add_subplot(projection='3d',) #gs[1, 1],
    for j in range(25):
        iztmp = -j
        section_yx = np.reshape(mx[iztmp,:,:],(ny,nx))
        xx_sec_yx,yy_sec_yx = np.meshgrid(vx,vy)
        zz_sec_yx = np.ones(xx_sec_yx.shape) * vz[iztmp]
        section_yx_clr = np.ones((ny*nx,4))*np.nan
        for i in range(len(valnotnan)):
            if labeltype=='facies':
                section_yx_clr[np.where(section_yx.flatten()==valnotnan[i]),:] = cmap(i)
            if labeltype=='age':
                section_yx_clr[np.where(section_yx.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
        section_yx_clr = np.reshape(section_yx_clr,(ny,nx,4))        
        surf = ax.plot_surface(xx_sec_yx,yy_sec_yx,zz_sec_yx, facecolors=section_yx_clr) #, cmap=cmap
    
    ixtmp = 0
    section_zy = np.reshape(mx[:,:,ixtmp],(nz,ny))
    yy_sec_zy,zz_sec_zy = np.meshgrid(vy,vz)
    xx_sec_zy = np.ones(yy_sec_zy.shape) * vx[ixtmp]
    section_zy_clr = np.ones((nz*ny,4))*np.nan
    for i in range(len(valnotnan)):
        if labeltype=='facies':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i)
        if labeltype=='age':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
    section_zy_clr = np.reshape(section_zy_clr,(nz,ny,4)) 
    surf = ax.plot_surface(xx_sec_zy,yy_sec_zy,zz_sec_zy, facecolors=section_zy_clr) #, cmap=cmap
    
    iytmp = 0
    section_zx = np.reshape(mx[:,iytmp,:],(nz,nx))
    xx_sec_zx,zz_sec_zx = np.meshgrid(vx,vz)
    yy_sec_zx = np.ones(xx_sec_zx.shape) * vy[iytmp]
    section_zx_clr = np.ones((nz*nx,4))*np.nan
    for i in range(len(valnotnan)):
        if labeltype=='facies':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(i)
        if labeltype=='age':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
    section_zx_clr = np.reshape(section_zx_clr,(nz,nx,4)) 
    surf = ax.plot_surface(xx_sec_zx,yy_sec_zx,zz_sec_zx, facecolors=section_zx_clr) #, cmap=cmap
    
    ixtmp = -1
    section_zy = np.reshape(mx[:,:,ixtmp],(nz,ny))
    yy_sec_zy,zz_sec_zy = np.meshgrid(vy,vz)
    xx_sec_zy = np.ones(yy_sec_zy.shape) * vx[ixtmp]
    section_zy_clr = np.ones((nz*ny,4))*np.nan
    for i in range(len(valnotnan)):
        section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i)
        if labeltype=='facies':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i)
        if labeltype=='age':
            section_zy_clr[np.where(section_zy.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
    section_zy_clr = np.reshape(section_zy_clr,(nz,ny,4)) 
    surf = ax.plot_surface(xx_sec_zy,yy_sec_zy,zz_sec_zy, facecolors=section_zy_clr) #, cmap=cmap
    
    iytmp = -1
    section_zx = np.reshape(mx[:,iytmp,:],(nz,nx))
    xx_sec_zx,zz_sec_zx = np.meshgrid(vx,vz)
    yy_sec_zx = np.ones(xx_sec_zx.shape) * vy[iytmp]
    section_zx_clr = np.ones((nz*nx,4))*np.nan
    for i in range(len(valnotnan)):
        if labeltype=='facies':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(i)
        if labeltype=='age':
            section_zx_clr[np.where(section_zx.flatten()==valnotnan[i]),:] = cmap(i/valnotnan.max())
    section_zx_clr = np.reshape(section_zx_clr,(nz,nx,4)) 
    surf = ax.plot_surface(xx_sec_zx,yy_sec_zx,zz_sec_zx,cmap=cmap, facecolors=section_zx_clr) #, cmap=cmap
    
    ax.set_aspect('equalxy')
    ax.view_init(25, -50, 0)
    ax.grid(False)

    if axisOff==False:
        ax.ticklabel_format(style='sci', axis='x', scilimits=(scilim_x,scilim_x))
        ax.ticklabel_format(style='sci', axis='y', scilimits=(scilim_y,scilim_y))
        ax.set_xlabel('Easting [m]'),ax.set_ylabel('Northing [m]'),ax.set_zlabel('Elevation [m]')
        if labeltype=='facies': ax.set_title('3D stochastic facies model',pad=-5)
        if labeltype=='age': ax.set_title('3D relative age model',pad=-5)
        
        cbar = fig.colorbar(surf, shrink=0.6,orientation='horizontal')
        if labeltype=='facies':
            cbar.ax.set_xticks((1/2+np.arange(len(valnotnan)))/(valnotnan.max()-valnotnan.min()+1))
            cbar.ax.set_xticklabels(['basement', 'weathered-b', 'intermediate','coarse-grained','fine-grained'])  # vertically oriented colorbar  rotation = 45
        if labeltype=='age':
            cbar.xlabel('relative age')
    else:
        ax.axis('off')
            
    
    if figFileName is not None:
        plt.savefig(figFileName,bbox_inches='tight')
    plt.show()
    return 

def plot_clay_and_sand_ecdfs_err(ecdf_clay_ref,ecdf_sand_ref,x01_vec,ecdf01_clay,ecdf01_sand,figsize=None,fontsize=None,figFileName=None):
    clay_ecdf_area_diff,clay_x01_1,clay_ecdf_1,clay_e1,clay_x01_2,clay_ecdf_2,clay_e2 = get_ecdf_area_difference(
        x01_vec,ecdf_clay_ref,x01_vec,ecdf01_clay) #/ ecdf_clay_area_err_mean_p5_p95[0]
    sand_ecdf_area_diff,sand_x01_1,sand_ecdf_1,sand_e1,sand_x01_2,sand_ecdf_2,sand_e2 = get_ecdf_area_difference(
        x01_vec,ecdf_sand_ref,x01_vec,ecdf01_sand) #/ ecdf_sand_area_err_mean_p5_p95[0]
    if figsize is None: figsize = (24/2.54,8/2.54)
    if fontsize is None: fontsize = 10
    if fontsize/figsize[0]>1.3: 
        br=True 
    else: 
        br=False
    
    label_0='difference'
    label_1='LBD reference'
    label_2='calibrated model'
    
    plt.rc("axes", linewidth=0.5) # so lines on edges of plots aren't too thick
    plt.matplotlib.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'}) # so that Arial is used
    plt.rcParams.update({'font.size': fontsize})    # size 10 font

    plt.subplots(1,2,dpi=300,figsize=figsize,sharey=True)

    if br==True:
        title_1 = '(a) Fine-grained sed. thickness\nECDFs difference: '+str(np.round(clay_ecdf_area_diff*100,1))+'%'
        title_2 = '(b) Coarse-grained sed. thickness\nECDFs difference: '+str(np.round(sand_ecdf_area_diff*100,1))+'%'
        ylbl = 'empirical cumulative\ndistribution function'
    else:
        title_1 = '(a) Fine-grained sed. thickness ECDFs difference: '+str(np.round(clay_ecdf_area_diff*100,1))+'%'
        title_2 = '(b) Coarse-grained sed. thickness ECDFs difference: '+str(np.round(sand_ecdf_area_diff*100,1))+'%'
        ylbl = 'empirical cumulative distribution function' 
    
    plt.subplot(121),plt.title(title_1)
    plt.tick_params(which='both',direction='inout', width=0.5)
    plt.fill_between(x01_vec,clay_e1,clay_e2,color='pink',label=label_0)
    plt.plot(clay_x01_1, clay_ecdf_1,'--k',label=label_1,linewidth=1)
    plt.plot(clay_x01_2, clay_ecdf_2,'-',color='gray',label=label_2,linewidth=.5) #orange
    plt.xlabel('normalized x')
    plt.ylabel(ylbl)
    plt.xlim([0,1])
    # plt.legend()
    
    plt.subplot(122),plt.title(title_2)
    plt.tick_params(which='both',direction='inout', width=0.5)
    plt.fill_between(x01_vec,sand_e1,sand_e2,color='pink',label=label_0)
    plt.plot(sand_x01_1, sand_ecdf_1,'--k',label=label_1,linewidth=1)
    plt.plot(sand_x01_2, sand_ecdf_2,'-',color='gray',label=label_2,linewidth=.5) #orange
    plt.xlabel('normalized x')
    # plt.ylabel('empirical cumulative distribution function')
    plt.xlim([0,1])
    plt.legend()

    if figFileName is not None:
        plt.savefig(figFileName,bbox_inches='tight')
    plt.show()
    return

def gen_concatenated_lenses_params(lenseClassesList,mask,dx,dz,seed):
    Lix,Liy,Liz,Ldipdir,Ldip,La,Lb,Lc,Ltruncr,Lfacies = [],[],[],[],[],[],[],[],[],[]
    for i in range(len(lenseClassesList)):
        ix,iy,iz,dipdir,dip,a,b,c,truncr = gen_lense_params(lenseClassesList[i],mask==1,dx,dz,seed+i)
        facies = np.ones(len(ix))*lenseClassesList[i].facies
        Lix = np.hstack((Lix,ix))
        Liy = np.hstack((Liy,iy))
        Liz = np.hstack((Liz,iz))
        Ldipdir = np.hstack((Ldipdir,dipdir))
        Ldip = np.hstack((Ldip,dip))
        La = np.hstack((La,a))
        Lb = np.hstack((Lb,b))
        Lc = np.hstack((Lc,c))
        Ltruncr = np.hstack((Ltruncr,truncr))
        Lfacies = np.hstack((Lfacies,facies))       
    return Lix.astype(int),Liy.astype(int),Liz.astype(int),Ldipdir,Ldip,La,Lb,Lc,Ltruncr,Lfacies.astype(int)

def gen_facies_youth_mx(vx,vy,vz,ix,iy,iz,dipdir,dip,a,b,c,truncr,facies,mask):
    nz = len(vz)
    ny = len(vy)
    nx = len(vx)
    zzz = np.tile(np.reshape(vz,(nz,1,1)),(1,ny,nx))
    topz = vz[iz] -c*truncr
    sort_idx = np.argsort(topz)
    # INIT
    relative_youth = mask+0
    facies_mx = mask+0
    rel_youth = 0
    relative_youth_z = np.zeros(len(facies))+np.nan
    # FACIES MATRIX
    for i in sort_idx:
        _,izyx = generate_reggrid_truncated_ellipsoid(vx,vy,vz,ix[i],iy[i],iz[i],
                                                      dipdir[i],dip[i],a[i],b[i],c[i],truncr[i])
        facies_mx[izyx] = facies[i]
        relative_youth[izyx] = rel_youth+1
        izarray = np.asarray(izyx)[0,:]
        if len(izarray):
            zidx = izarray.max()
            relative_youth_z[rel_youth] = vz[zidx]
        rel_youth += 1
        del izyx,izarray
    facies_mx = facies_mx * mask
    # RELATIVE YOUTH MATRIX
    relative_youth_mx = np.zeros((nz,ny,nx))
    for i in range(rel_youth):
        if np.isnan(relative_youth_z[i]):
            continue
        else:
            relative_youth_mx[np.where(zzz>relative_youth_z[i])]=i+1
    relative_youth_mx[np.where(facies_mx>1)] = relative_youth[np.where(facies_mx>1)]
    relative_youth_mx = relative_youth_mx * mask
   
    return facies_mx, relative_youth_mx#, facies_cnt

def get_facies_prop(mx,mask,lithocodes,labels):
    df = pd.DataFrame(columns=['lithocodes','count','proportion','labels'])
    df['lithocodes'] = lithocodes
    df['labels'] = labels
    df.set_index('lithocodes',inplace=True)
    for i in lithocodes:
        cnt = np.sum((mx.flatten()==i)*1)
        df.loc[i,'count'] = cnt
    df['proportion'] = np.around((100*df['count'].values.flatten()/(df['count'].sum())).astype(float),2)
    return df

def get_model_dh_litho_thickness(facies_mx,ix,iy,lithocodes,zmin,dz):
    bhid = 'ix'+str(ix)+'iy'+str(iy)
    dh_facies = facies_mx[:,iy,ix]+0
    dh_facies[np.isnan(dh_facies)]=np.max(lithocodes)+1
    diff = np.concatenate((np.array([False]),np.abs(np.diff(dh_facies))>0))*1
    ix_btm = np.where(diff==1)
    ix_top = np.asarray(ix_btm)-1
    ix_btm = np.concatenate((np.array([0]),np.asarray(ix_btm).flatten()))
    ix_top = np.concatenate((ix_top.flatten(),np.array([len(diff)-1])))
    lithocode = dh_facies[ix_btm]
    thickness = (ix_top - ix_btm + 1) * dz
    ztop = zmin + dz/2 + dz * ix_top
    zbtm = zmin + dz/2 + dz * ix_btm
    df = pd.DataFrame(columns=['bhid','from','to','lithocode','thickness'])
    df['from'] = ztop
    df['to'] = zbtm
    df['lithocode'] = lithocode
    df['thickness'] = thickness
    df['bhid'] = bhid
    return df

def get_model_litho_thickness(facies_mx,ndhsamples,nx,ny,lithocodes,zmin,dz,rng):
    df_model_thickness = pd.DataFrame(columns=['bhid','from','to','lithocode','thickness'])
    tmp = rng.random((ndhsamples,2))
    ix2sample = np.floor(tmp[:,0]*nx).astype(int)
    iy2sample = np.floor(tmp[:,1]*ny).astype(int)
    for i in range(ndhsamples):
        tmp_df = get_model_dh_litho_thickness(facies_mx,ix2sample[i],iy2sample[i],lithocodes,zmin,dz)
        if not tmp_df.empty:
            df_model_thickness = pd.concat([df_model_thickness if not df_model_thickness.empty else None, tmp_df])

    df_model_thickness.reset_index(drop=True, inplace=True)
    return df_model_thickness

def get_level_0_elem_ecdf(df,elem):
    ix_elem = np.asarray(np.where((df['level_0']==elem)&(df['thickness'].isnull()==False)&(df['thickness']>0))).flatten()
    nx = len(ix_elem)
    x = np.concatenate([np.array([0]),np.sort(df.loc[ix_elem,'thickness'].values)])
    ecdf = np.linspace(0,nx,num=(nx+1),endpoint=True)/nx
    return x, ecdf

def get_summary_stats_level_0(df_processed_log3):
    ix_sediments = np.asarray(np.where(
        (df_processed_log3['level_0']!='Basement') & 
        (df_processed_log3['level_0']!='Weathered basement')
    )).flatten()
    
    df_stats_by_level = df_processed_log3.loc[ix_sediments,['level_0','level_1','thickness']].groupby(['level_0','level_1'],as_index=False).sum()
    df_stats_by_level['proportion']=df_stats_by_level['thickness']/df_stats_by_level['thickness'].sum()
    df_stats_level_0 = df_stats_by_level.loc[:,['level_0','proportion']].groupby(['level_0'],as_index=False).sum()
    df_stats_level_0.set_index('level_0',inplace=True,drop=True)
    
    df = df_processed_log3[['level_0','level_1','thickness']]
    x_clay, ecdf_clay = get_level_0_elem_ecdf(df,'Clay')
    x_sand, ecdf_sand = get_level_0_elem_ecdf(df,'Sands')
    x_inte, ecdf_inte = get_level_0_elem_ecdf(df,'Intermediate') 
    
    return ix_sediments, df_stats_level_0,x_clay, ecdf_clay, x_sand, ecdf_sand, x_inte, ecdf_inte

def plot_summary_stats_level_0(df_processed_log3,thickness_lim,figsize=None,fontsize=None,figFileName=None):
    ix_sediments, df_stats_level_0,x_clay, ecdf_clay, x_sand, ecdf_sand, x_inte, ecdf_inte = get_summary_stats_level_0(df_processed_log3)
    if figsize is None: figsize = (15/2.54,8/2.54)
    if fontsize is None: fontsize = 5

    plt.rc("axes", linewidth=0.5) # so lines on edges of plots aren't too thick
    plt.matplotlib.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'}) # so that Arial is used
    plt.rcParams.update({'font.size': fontsize})    # size 10 font
    
    plt.subplots(2,2,dpi=300,figsize=figsize)
    plt.subplot(221),plt.title('(a) Sediment proportions')
    # plt.pie(df_stats_level_0.loc[['Intermediate','Clay','Sands'],'proportion'], labels=['Intermediate','Clay','Sands'], autopct='%1.1f%%')
    plt.pie(df_stats_level_0.loc[['Intermediate','Sands','Clay'],'proportion'], labels=['Intermediate','Coarse-grained','Fine-grained'],colors = ['sienna', 'gold', 'grey'], autopct='%1.1f%%')
    
    plt.subplot(223),plt.title('(b) Thickness distribution')
    plt.grid(False)
    plt.tick_params(which='both',direction='inout', width=0.5)
    ax=sns.violinplot(data=df_processed_log3.loc[ix_sediments,['level_0','level_1','thickness']],x='level_0',order=["Intermediate","Sands","Clay"],
                   y="thickness",hue ='level_0',legend=False,palette = ['sienna', 'gold', 'grey'],hue_order=["Intermediate","Sands","Clay"]) #,x='level_0' boxplot violinplot
    ax.set_xticklabels(['Intermediate','\n Coarse-grained','Fine-grained'])
    plt.ylim([0,10])
    plt.xlabel('')
    plt.ylabel('thickness [m]')
    
    plt.subplot(222),plt.axis('off')
    plt.subplot(224),plt.axis('off')
    
    plt.subplot(122),plt.title('(c) Empirical cumulative distributions')
    plt.tick_params(which='both',direction='inout', width=0.5)
    plt.grid(False)
    plt.xlim(thickness_lim)
    plt.plot(x_inte, ecdf_inte,label='Intermediate',color='sienna')
    plt.plot(x_sand, ecdf_sand,label='Coarse-grained',color='gold')
    plt.plot(x_clay, ecdf_clay,'--',label='Fine-grained',color='grey')
    plt.xlabel('thickness [m]')
    plt.ylabel('cumulative distribution')
    plt.legend()

    ftfgsfactor = 3 *fontsize/figsize[0]/2.54
    # print('ftfgsfactor: '+str(ftfgsfactor))
    plt.subplots_adjust(wspace = 0.2 * ftfgsfactor, hspace = 0.1* ftfgsfactor)
    
    if figFileName is not None:
        plt.savefig(figFileName,bbox_inches='tight')
    plt.show()
    return
    
def compute_err(ref_err_fn,calib_model_file,ndhsamples,seed):
    rng = np.random.default_rng(seed)
    f = open(ref_err_fn, 'rb')
    [prop_clay_mean_p5_p95, prop_sand_mean_p5_p95, prop_inte_mean_p5_p95,
     prop_clay_err_mean_p5_p95, prop_sand_err_mean_p5_p95, prop_inte_err_mean_p5_p95,
     ecdf_clay_mean_p5_p95, ecdf_sand_mean_p5_p95, ecdf_inte_mean_p5_p95, x01_vec, thickness_lim,
     ecdf_clay_area_err_mean_p5_p95, ecdf_sand_area_err_mean_p5_p95, ecdf_inte_area_err_mean_p5_p95] = pickle.load(f)
    f.close()
    nbins = len(x01_vec)-1
    xlim = thickness_lim[1]
    ecdf_clay_ref = ecdf_clay_mean_p5_p95[:,0]
    ecdf_sand_ref = ecdf_sand_mean_p5_p95[:,0]

    npzfile = np.load(calib_model_file)
    facies_mx = npzfile['facies_mx']
    labels = npzfile['labels']
    lithocodes = npzfile['lithocodes']
    lithocolors = npzfile['lithocolors']
    mask_aquifer = npzfile['mask_aquifer']
    relative_youth_mx = npzfile['relative_youth_mx']
    vx = npzfile['vx']
    vy = npzfile['vy']
    vz = npzfile['vz']    
    dz = vz[1]-vz[0]
    zmin = vz.min()-dz/2
    
    nz,ny,nx = facies_mx.shape
    # COMPUTE COUNT AND PROPORTIONS PER LITHOCODE
    #printtimelog('compute proportions')
    prop = get_facies_prop(facies_mx,mask_aquifer,lithocodes[2:],labels[2:])
    #print(prop)
    
    # GET THICKNESSES
    #printtimelog('extract thicknesses')
    df_model_thickness = get_model_litho_thickness(facies_mx,ndhsamples,nx,ny,lithocodes,zmin,dz,rng)
    
    # COMPUTE EMPIRICAL CUMULATIVE DISTRIBUTION FUNCTIONS OF THICKNESS PER LITHOCODE
    #printtimelogwithstarline('COMPUTE ECDFs')
    x_clay, ecdf_clay = get_ecdf(df_model_thickness,3)
    x_sand, ecdf_sand = get_ecdf(df_model_thickness,2)
    x_inte, ecdf_inte = get_ecdf(df_model_thickness,1)
    
    #printtimelog('normalize ecdfs')
    
    ecdf01_clay = interp_ecdf_normalize_x(x_clay,ecdf_clay,xlim,x01_vec)
    ecdf01_sand = interp_ecdf_normalize_x(x_sand,ecdf_sand,xlim,x01_vec)
    ecdf01_inte = interp_ecdf_normalize_x(x_inte,ecdf_inte,xlim,x01_vec)
    
    #printtimelogwithstarline('COMPUTE DISTANCE TO LBD DATASET')
    #printtimelog('compute errors')
    err_prop_clay = (float(prop.loc[3,'proportion'])/100-prop_clay_mean_p5_p95[0]) # / prop_clay_err_mean_p5_p95[0]
    err_prop_sand = (float(prop.loc[2,'proportion'])/100-prop_sand_mean_p5_p95[0]) #/ prop_sand_err_mean_p5_p95[0]
    
    
    err_ecfd_clay,_,_,_,_,_,_ = get_ecdf_area_difference(x01_vec,ecdf_clay_ref,x01_vec,ecdf01_clay,nbins=nbins,plot=False) #/ ecdf_clay_area_err_mean_p5_p95[0]
    err_ecfd_sand,_,_,_,_,_,_ = get_ecdf_area_difference(x01_vec,ecdf_sand_ref,x01_vec,ecdf01_sand,nbins=nbins,plot=False) #/ ecdf_sand_area_err_mean_p5_p95[0]
    # err_ratio_ecfd_inte,_,_,_,_,_, = get_ecdf_area_difference(x01_vec,ecdf_inte_mean_p5_p95[:,0],x01_vec,ecdf01_inte,nbins=nbins) #/ ecdf_inte_area_err_mean_p5_p95[0]
    
    #printtimelog('compute distance')
    errors = np.abs(np.array([err_prop_clay,err_prop_sand,err_ecfd_clay,err_ecfd_sand]))
    print('errors: '+str(errors))
    
    similarity = np.prod(1-errors)**(1/len(errors))
    dist2lbd = 1 - similarity
    print('dist2lbd: '+str(dist2lbd))
    return x01_vec,ecdf01_clay,ecdf01_sand,ecdf_clay_ref,ecdf_sand_ref,facies_mx,relative_youth_mx,lithocolors,vx,vy,vz

def plot_spatial_prop(df_2D_prop,figsize=None,fontsize=None,figFileName=None):
    scilim_x = 5
    scilim_y = 6
    
    if figsize is None: figsize = (15/2.54,5/2.54)
    if fontsize is None: fontsize = 5

    plt.rc("axes", linewidth=0.5) # so lines on edges of plots aren't too thick
    plt.matplotlib.rc('font', **{'sans-serif' : 'Arial','family' : 'sans-serif'}) # so that Arial is used
    plt.rcParams.update({'font.size': fontsize})    # size 10 font
   
    s=4
    
    fig,ax = plt.subplots(1,4,dpi=300,sharey=True,figsize=figsize,width_ratios=[1,1,1,1/4])
    ax[0].set_title('Fine-grained\nsediment %')
    im0=ax[0].scatter(df_2D_prop['Easting'].values,
                      df_2D_prop['Northing'].values,
                      c = df_2D_prop['prop_clay'].values*1E2,
                      cmap='viridis',s=s)
    ax[0].set_aspect(1)
    ax[0].ticklabel_format(style='sci', axis='x', scilimits=(scilim_x,scilim_x))
    ax[0].ticklabel_format(style='sci', axis='y', scilimits=(scilim_y,scilim_y))
    ax[0].set_xlabel('Easting [m]')
    ax[0].set_ylabel('Northing [m]')
    # fig.colorbar(im0,ax=ax[0],orientation='horizontal', fraction=.035)
    
    ax[1].set_title('Coarse-grained\nsediment %')
    im1=ax[1].scatter(df_2D_prop['Easting'].values,
                      df_2D_prop['Northing'].values,
                      c = df_2D_prop['prop_sand'].values*1E2,
                      cmap='viridis',s=s)
    ax[1].set_aspect(1)
    ax[1].ticklabel_format(style='sci', axis='x', scilimits=(scilim_x,scilim_x))
    ax[1].ticklabel_format(style='sci', axis='y', scilimits=(scilim_y,scilim_y))
    ax[1].set_xlabel('Easting [m]')
    # ax[1].set_ylabel('Northing [m]')
    # fig.colorbar(im1,ax=ax[1],orientation='horizontal', fraction=.035)
    
    ax[2].set_title('Intermediate\nsediment %')
    im2=ax[2].scatter(df_2D_prop['Easting'].values,
                      df_2D_prop['Northing'].values,
                      c = df_2D_prop['prop_inte'].values*1E2,
                      cmap='viridis',s=s)
    ax[2].set_aspect(1)
    ax[2].ticklabel_format(style='sci', axis='x', scilimits=(scilim_x,scilim_x))
    ax[2].ticklabel_format(style='sci', axis='y', scilimits=(scilim_y,scilim_y))
    ax[2].set_xlabel('Easting [m]')
    # ax[2].set_ylabel('Northing [m]')
    # fig.colorbar(im2,ax=ax[2],orientation='horizontal', fraction=.035)
    ax[3].axis('off')
    fig.colorbar(im0,ax=ax[3], fraction=.5) #images[0], 

    if figFileName is not None:
        plt.savefig(figFileName,bbox_inches='tight')
    plt.show()
    return
