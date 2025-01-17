# -*- coding: utf-8 -*-
"""
Created on Tue Nov 14 08:46:01 2023

@author: Guillaume PIROT
"""
import numpy as np
import geone as gn
from utils import get_plane_coeff

def gen_grf_planar_trend(vx,vy,pts,cov_model,seed):
    nx = len(vx)
    ny = len(vy)
    dx = vx[1]-vx[0]
    dy = vy[1]-vy[0]
    ox = vx[0]
    oy = vy[0]
    dimension = (nx, ny)
    spacing = (dx, dy)
    origin = (ox, oy)
    yy,xx = np.meshgrid(vy,vx,indexing='ij')
    # planar trend from three points
    a,b,c,d = get_plane_coeff(pts)
    zz = -(d+b*yy+a*xx)/c
    # generate Gaussian Random Field
    np.random.seed(seed)
    sim2Da = gn.grf.grf2D(cov_model, dimension, spacing, origin, nreal=1)
    im2a = gn.img.Img(nx, ny, 1, dx, dy, 1., ox, oy, 0., nv=1, val=sim2Da)
    grf = np.reshape(im2a.val[0,0,:,:],(ny,nx))
    return grf+zz
