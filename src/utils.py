# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:46:01 2023

@author: Guillaume PIROT
"""

import numpy as np
from scipy.interpolate import interp1d 
import scipy.linalg
import matplotlib.pyplot as plt


def get_plane_coeff(pts):
    # 1 col per point in pts array
    # Define planar vectors
    pq = pts[:,1]-pts[:,0]
    pr = pts[:,2]-pts[:,0]
    # point P
    p =  pts[:,0]
    # normal to the plane
    n = np.cross(pq,pr)
    # plane equation (ax+by+cz+d=0) coeffs
    a = n[0]
    b = n[1]
    c = n[2]
    d = -(np.dot(n,p))  
    return a,b,c,d         

def rotZ(vector_ini,theta):
    rotZmx = np.asarray([[np.cos(theta),-np.sin(theta),0],
                         [np.sin(theta),np.cos(theta),0],
                         [0,0,1]])
    return np.matmul(rotZmx,vector_ini)

def rotY(vector_ini,theta):
    rotZmx = np.asarray([[np.cos(theta),0,np.sin(theta)],
                         [0,1,0],
                         [-np.sin(theta),0,np.cos(theta)]])
    return np.matmul(rotZmx,vector_ini)

def check_plot_2_sections(mx1,mx2,title1,title2,cmap='inferno'):
    fig,ax = plt.subplots(1,2,dpi=100)
    ax[0].set_title(title1)
    im0=ax[0].imshow(mx1, origin='lower' ,cmap=cmap)
    fig.colorbar(im0,ax=ax[0],orientation='horizontal', fraction=.035)
    ax[1].set_title(title2)
    im1=ax[1].imshow(mx2, origin='lower',cmap=cmap)
    fig.colorbar(im1,ax=ax[1],orientation='horizontal', fraction=.035)
    plt.show()
    return

def get_ecdf(df,lithocode=None):
    if lithocode is not None:
        ix_elem = np.asarray(np.where((df['lithocode']==lithocode)&(df['thickness'].isnull()==False)&(df['thickness']>0))).flatten()
    else:
        ix_elem = np.arange(len(df))
    nx = len(ix_elem)
    x = np.concatenate([np.array([0]),np.sort(df.loc[ix_elem,'thickness'].values)])
    ecdf = np.linspace(0,nx,num=(nx+1),endpoint=True)/nx
    return x, ecdf


def get_strictly_increasing_x_indices(x):
    _, indices_x = np.unique(x, return_index=True)
    indices_x = np.concatenate(((indices_x[1:]-1).flatten(),np.array(len(x)-1).flatten()))
    return indices_x    

def interpolate_ecdf(x01,ecdf,xi):
    # x01 should be the normalised vector of the x variable between [0,1]
    # xi: where to interpolate
    # get strictly increasing x
    if x01.max()<1:
        x01 = np.concatenate((x01,np.array([1])))
        ecdf = np.concatenate((ecdf,np.array([1])))
    indices = get_strictly_increasing_x_indices(x01)
    # interpolate cdf for normalized_thickness in [0,1]    
    cs = interp1d(x01[indices], ecdf[indices],kind='linear')
    return cs(xi)

def interp_ecdf_normalize_x(x,ecdf,xlim,x01_vec):
    x01 = x/xlim
    ecdf01 = interpolate_ecdf(x01,ecdf,x01_vec)
    return ecdf01

def get_ecdf_area_difference(x_1,ecdf_1,x_2,ecdf_2,nbins=100,title=None,label_1=None,label_2=None,xlim=None,plot=False,verb=False):
    
    if label_1 is None: label_1='ecdf_1'
    if label_2 is None: label_2='ecdf_2'
    if xlim is None:
        xmin = np.min([x_1.min(),x_2.min()])
        xlim = np.max([x_1.max(),x_2.max()]) - xmin
    else:
        xmin = 0
        # ix2keep_1 = np.asarray(np.where(x_1<xlim)).flatten()
        # ecdf_1 = np.concatenate((ecdf_1[ix2keep_1].flatten(),np.array([1.0]).flatten()))
        # x_1 = np.concatenate((x_1[ix2keep_1].flatten(),np.array([1.0]).flatten()))
        # ix2keep_2 = np.asarray(np.where(x_2<xlim)).flatten()
        # ecdf_2 = np.concatenate((ecdf_2[ix2keep_2].flatten(),np.array([1.0]).flatten()))
        # x_2 = np.concatenate((x_2[ix2keep_2].flatten(),np.array([1.0]).flatten()))
    # Normalize x between 0 and 1
    x01_1 = (x_1-xmin)/xlim
    x01_2 = (x_2-xmin)/xlim
    # intorpolate at regular intervals
    x01_vec = np.linspace(0,1.0,num=(nbins+1), endpoint=True)
    e1 = interpolate_ecdf(x01_1,ecdf_1,x01_vec) 
    e2 = interpolate_ecdf(x01_2,ecdf_2,x01_vec) 
    # compute area difference
    ecdf_area_diff = np.sum(np.abs( e1 - e2))/nbins
    if title is None: title='Area difference between 2 ECDFs: '+str(np.round(ecdf_area_diff*100,2))+'%'
    else: title=title+' difference: '+str(np.round(ecdf_area_diff*100,2))+'%'
    if verb: print(' ecdf_area_diff: '+str(ecdf_area_diff))
    if plot is not False:
        plt.figure(dpi=300)
        plt.fill_between(x01_vec,e1,e2,color='pink')
        plt.plot(x01_1, ecdf_1,'--k',label=label_1,linewidth=1)
        plt.plot(x01_2, ecdf_2,'-',color='gray',label=label_2,linewidth=.5) #orange
        plt.title(title)
        plt.xlabel('normalized x')
        plt.ylabel('empirical cumulative distribution function')
        plt.xlim([0,1])
        plt.legend()
        # plt.text(0.4,0.1,'Area difference between \nthe two ecdfs: '+str(np.round(ecdf_area_diff*100,2))+'%')
        plt.show()
    return ecdf_area_diff,x01_1,ecdf_1,e1,x01_2,ecdf_2,e2

def get_ecdf_area_difference_old(x_1,ecdf_1,x_2,ecdf_2,nbins=100,title=None,label_1=None,label_2=None,xlim=None,plot=False,verb=False):
    if title is None: title='Area difference between 2 ECDFs'
    if label_1 is None: label_1='ecdf_1'
    if label_2 is None: label_2='ecdf_2'
    if xlim is None:
        xmin = np.min([x_1.min(),x_2.min()])
        xlim = np.max([x_1.max(),x_2.max()]) - xmin
    else:
        xmin = 0
        # ix2keep_1 = np.asarray(np.where(x_1<xlim)).flatten()
        # ecdf_1 = np.concatenate((ecdf_1[ix2keep_1].flatten(),np.array([1.0]).flatten()))
        # x_1 = np.concatenate((x_1[ix2keep_1].flatten(),np.array([1.0]).flatten()))
        # ix2keep_2 = np.asarray(np.where(x_2<xlim)).flatten()
        # ecdf_2 = np.concatenate((ecdf_2[ix2keep_2].flatten(),np.array([1.0]).flatten()))
        # x_2 = np.concatenate((x_2[ix2keep_2].flatten(),np.array([1.0]).flatten()))
    # Normalize x between 0 and 1
    x01_1 = (x_1-xmin)/xlim
    x01_2 = (x_2-xmin)/xlim
    # intorpolate at regular intervals
    x01_vec = np.linspace(0,1.0,num=(nbins+1), endpoint=True)
    e1 = interpolate_ecdf(x01_1,ecdf_1,x01_vec) 
    e2 = interpolate_ecdf(x01_2,ecdf_2,x01_vec) 
    # compute area difference
    ecdf_area_diff = np.sum(np.abs( e1 - e2))/nbins
    if verb: print(' ecdf_area_diff: '+str(ecdf_area_diff))
    if plot:
        plt.figure(dpi=300)
        plt.fill_between(x01_vec,e1,e2)
        plt.plot(x01_1, ecdf_1,'--k',label=label_1)
        plt.plot(x01_2, ecdf_2,'--',color='orange',label=label_2)
        plt.title(title)
        plt.xlabel('normalized x')
        plt.ylabel('empirical cumulative distribution function')
        plt.xlim([0,1])
        plt.legend()
        plt.text(0.4,0.1,'ecdf_area_diff: '+str(np.round(ecdf_area_diff*100,2))+'%')
        plt.show()
    return ecdf_area_diff

def get_spherical_coordinate_from_proportions(p1,p2,p3,verb=False):
    if verb: print('**********************************************************************')
    # get third proporion
    r = 1
    #p3 = r-p1-p2
    if verb: print('p1: '+str(p1))
    if verb: print('p2: '+str(p2))
    if verb: print('p3: '+str(p3))
    # transform to x,y,z coord 
    psum = (p1+p2+p3)
    x = np.sqrt(p1/psum)
    y = np.sqrt(p2/psum)
    z = np.sqrt(p3/psum)
    if verb: print('x: '+str(x))
    if verb: print('y: '+str(y))
    if verb: print('z: '+str(z))
    # transform to spherical coordinales
    theta = np.arccos(z/r) # in radians
    phi = np.sign(y)*np.arccos(x/np.sqrt(x**2+y**2+1*((x==0)&(y==0)))) # in radians
    if verb: print('theta: '+str(theta) +' radians') 
    if verb: print('phi: '+str(phi) +' radians\n')
    return theta, phi

def get_proportions_from_spherical_coordinates(theta,phi,verb=False):
    if verb: print('**********************************************************************')
    r = 1.0
    x = r * np.sin(theta)*np.cos(phi)
    y = r * np.sin(theta)*np.sin(phi)
#     z = r * np.cos(theta)
    if verb: print('x: '+str(x))
    if verb: print('y: '+str(y))
#     if verb: print('z: '+str(z))
    p1 = x**2
    p2 = y**2
    p3 = r - x**2 - y**2
    if verb: print('p1: '+str(p1))
    if verb: print('p2: '+str(p2))
    if verb: print('p3: '+str(p3)+'\n')
    return p1,p2,p3

# p1 = 0.35
# p2 = 0.32
# theta, phi = get_spherical_coordinate_from_proportions(p1,p2,1-(p1+p2),verb=True)
# p1back,p2back,p3back = get_proportions_from_spherical_coordinates(theta,phi,verb=True)
# get_spherical_coordinate_from_proportions(1,0,0,verb=True)
# get_spherical_coordinate_from_proportions(0,1,0,verb=True)
# get_spherical_coordinate_from_proportions(0,0,1,verb=True)

def get_best_fitting_plane(df_2D_prop,colname):
    # best-fit linear plane
    A = np.c_[df_2D_prop['Easting'].values, df_2D_prop['Northing'].values, np.ones(len(df_2D_prop))]
    plane_coeffs,_,_,_ = scipy.linalg.lstsq(A, df_2D_prop[colname].values)    # coefficients
    return plane_coeffs

def get_Z_from_plane_coeffs(X,Y,plane_coeffs):
    return plane_coeffs[0]*X + plane_coeffs[1]*Y + plane_coeffs[2]

def remove_trend_from_data(df_2D_prop,colname,plane_coeffs):
    val = df_2D_prop[colname].values - (plane_coeffs[0]*df_2D_prop['Easting'].values + 
                                             plane_coeffs[1]*df_2D_prop['Northing'].values + 
                                             plane_coeffs[2])
    return val



