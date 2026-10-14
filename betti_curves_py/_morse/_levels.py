"""Compute image-cubical filtration values without constructing cell objects."""
import numpy as np
from numba import njit

@njit(cache=True)
def _full_levels(v):
 nx,ny,nz=v.shape
 sy,sz=2*ny+1,2*nz+1
 levels=np.empty((2*nx+1)*sy*sz,dtype=np.int32)
 for x in range(2*nx+1):
  for y in range(sy):
   for z in range(sz):
    level=65536
    for dx in range(2):
     i=x//2 if x&1 else x//2-1+dx
     if i<0 or i>=nx:continue
     for dy in range(2):
      j=y//2 if y&1 else y//2-1+dy
      if j<0 or j>=ny:continue
      for dz in range(2):
       k=z//2 if z&1 else z//2-1+dz
       if k<0 or k>=nz:continue
       a=int(v[i,j,k]);level=min(level,a)
    levels[(x*sy+y)*sz+z]=level
 return levels

