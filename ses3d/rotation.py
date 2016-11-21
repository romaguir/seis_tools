# -*- coding: iso-8859-1 -*-
import numpy as np
import matplotlib.pylab as plt

###################################################################################################
#- rotation matrix
###################################################################################################

def rotation_matrix(n,phi):

  """ compute rotation matrix
  input: rotation angle phi [deg] and rotation vector n normalised to 1
  return: rotation matrix
  """

  phi=np.pi*phi/180.0

  A=np.array([ (n[0]*n[0],n[0]*n[1],n[0]*n[2]), (n[1]*n[0],n[1]*n[1],n[1]*n[2]), (n[2]*n[0],n[2]*n[1],n[2]*n[2])])
  B=np.eye(3)
  C=np.array([ (0.0,-n[2],n[1]), (n[2],0.0,-n[0]), (-n[1],n[0],0.0)])

  R=(1.0-np.cos(phi))*A+np.cos(phi)*B+np.sin(phi)*C

  return np.matrix(R)


###################################################################################################
#- rotate coordinates
###################################################################################################

def rotate_coordinates(n,phi,colat,lon):

  """ rotate colat and lon
  input: rotation angle phi [deg] and rotation vector n normalised to 1, original colatitude and longitude [deg]
  return: colat_new [deg], lon_new [deg]
  """

  # convert to radians

  colat=np.pi*colat/180.0
  lon=np.pi*lon/180.0

  # rotation matrix

  R=rotation_matrix(n,phi)

  # original position vector

  x=np.matrix([[np.cos(lon)*np.sin(colat)], [np.sin(lon)*np.sin(colat)], [np.cos(colat)]])

  # rotated position vector

  y=R*x

  # compute rotated colatitude and longitude

  colat_new=np.arccos(y[2])
  lon_new=np.arctan2(y[1],y[0])

  return float(180.0*colat_new/np.pi), float(180.0*lon_new/np.pi)


###################################################################################################
#- rotate moment tensor
###################################################################################################

def rotate_moment_tensor(n,phi,colat,lon,M):

  """ rotate moment tensor
  input: rotation angle phi [deg] and rotation vector n normalised to 1, original colat and lon [deg], original moment tensor M as matrix
  M=[Mtt Mtp Mtr
     Mtp Mpp Mpr
     Mtr Mpr Mrr]
  return: rotated moment tensor
  """

  # rotation matrix

  R=rotation_matrix(n,phi)

  # rotated coordinates

  colat_new,lon_new=rotate_coordinates(n,phi,colat,lon)

  # transform to radians

  colat=np.pi*colat/180.0
  lon=np.pi*lon/180.0

  colat_new=np.pi*colat_new/180.0
  lon_new=np.pi*lon_new/180.0

  # original basis vectors with respect to unit vectors [100].[010],[001]

  bt=np.matrix([[np.cos(lon)*np.cos(colat)],[np.sin(lon)*np.cos(colat)],[-np.sin(colat)]])
  bp=np.matrix([[-np.sin(lon)],[np.cos(lon)],[0.0]])
  br=np.matrix([[np.cos(lon)*np.sin(colat)],[np.sin(lon)*np.sin(colat)],[np.cos(colat)]])

  # original basis vectors with respect to rotated unit vectors

  bt=R*bt
  bp=R*bp
  br=R*br

  # new basis vectors with respect to rotated unit vectors

  bt_new=np.matrix([[np.cos(lon_new)*np.cos(colat_new)],[np.sin(lon_new)*np.cos(colat_new)],[-np.sin(colat_new)]])
  bp_new=np.matrix([[-np.sin(lon_new)],[np.cos(lon_new)],[0.0]])
  br_new=np.matrix([[np.cos(lon_new)*np.sin(colat_new)],[np.sin(lon_new)*np.sin(colat_new)],[np.cos(colat_new)]])

  # assemble transformation matrix and return

  A=np.matrix([[float(bt_new.transpose()*bt), float(bt_new.transpose()*bp), float(bt_new.transpose()*br)],[float(bp_new.transpose()*bt), float(bp_new.transpose()*bp), float(bp_new.transpose()*br)],[float(br_new.transpose()*bt), float(br_new.transpose()*bp), float(br_new.transpose()*br)]])

  return A*M*A.transpose()

