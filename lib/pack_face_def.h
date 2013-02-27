// The following indexing routines work for arbitrary (including odd) lattice dimensions.
// compute an index into the local volume from an index into the face (used by the face packing routines)

  template <int dim, int nLayers>
static inline __device__ int indexFromFaceIndex(int face_idx, const int &face_volume,
    const int &face_num, const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3; // face_T = X4;
  switch (dim) {
    case 0:
      face_X = nLayers;
      break;
    case 1:
      face_Y = nLayers;
      break;
    case 2:
      face_Z = nLayers;
      break;
    case 3:
      // face_T = nLayers;
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // intrinsic parity of the face depends on offset of first element

  int face_parity;
  switch (dim) {
    case 0:
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }

  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;

  if (!(face_X & 1)) { // face_X even
    //   int t = face_idx / face_XYZ;
    //   int z = (face_idx / face_XY) % face_Z;
    //   int y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + t + z + y) & 1;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    int aux2 = aux1 / face_Y;
    int y = aux1 - aux2 * face_Y;
    int t = aux2 / face_Z;
    int z = aux2 - t * face_Z;
    face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    int t = face_idx / face_XYZ;
    int z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
  } else if (!(face_Z & 1)) { // face_Z even
    int t = face_idx / face_XYZ;
    face_idx += (face_parity + t) & 1;
  } else {
    face_idx += face_parity;
  }

  // compute index into the full local volume

  int idx = face_idx;
  int gap, aux;

  switch (dim) {
    case 0:
      gap = X1 - nLayers;
      aux = face_idx / face_X;
      idx += (aux + face_num) * gap;
      break;
    case 1:
      gap = X2 - nLayers;
      aux = face_idx / face_XY;
      idx += (aux + face_num) * gap * face_X;
      break;
    case 2:
      gap = X3 - nLayers;
      aux = face_idx / face_XYZ;
      idx += (aux + face_num) * gap * face_XY;
      break;
    case 3:
      gap = X4 - nLayers;
      idx += face_num * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}

// compute an index into the local volume from an index into the face (used by the face packing routines)
// G.Shi: the spinor order in ghost region is different between wilson and asqtad, thus different index
//	  computing routine.
  template <int dim, int nLayers>
static inline __device__ int indexFromFaceIndexAsqtad(int face_idx, const int &face_volume,
    const int &face_num, const int &parity, int* x)
{
  // dimensions of the face (FIXME: optimize using constant cache)
  int dims[3];
  int V = x[0]*x[1]*x[2]*x[3];
  int face_X = x[0], face_Y = x[1], face_Z = x[2]; // face_T = X4;


  if(face_idx ==0){
    printf("In indexFromFaceIndexAsqtad\n");
    printf("X1 = %d, x[0] = %d\n", X1, x[0]);
    printf("X2 = %d, x[1] = %d\n", X2, x[1]);
    printf("X3 = %d, x[2] = %d\n", X3, x[2]);
    printf("X4 = %d, x[3] = %d\n", X4, x[3]);
    printf("V = %d\n",V);
  }

  switch (dim) {
    case 0:
      face_X = nLayers;
      dims[0]= x[1]; dims[1]=x[2]; dims[2]=x[3];
      break;
    case 1:
      face_Y = nLayers;
      dims[0]=x[0]; dims[1]=x[2]; dims[2]=x[3];
      break;
    case 2:
      face_Z = nLayers;
      dims[0]=x[0]; dims[1]=x[1]; dims[2]=x[3];
      break;
    case 3:
      // face_T = nLayers;
      dims[0]=x[0]; dims[1]=x[1]; dims[2]=x[3];
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // intrinsic parity of the face depends on offset of first element

  int face_parity;
  switch (dim) {
    case 0:
      face_parity = (parity + face_num * (x[0] - nLayers)) & 1;
      break;
    case 1:
      face_parity = (parity + face_num * (x[1] - nLayers)) & 1;
      break;
    case 2:
      face_parity = (parity + face_num * (x[2] - nLayers)) & 1;
      break;
    case 3:
      face_parity = (parity + face_num * (x[3] - nLayers)) & 1;
      break;
  }


  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;
  /*y,z,t here are face indexes in new order*/
  int aux1 = face_idx / dims[0];
  int aux2 = aux1 / dims[1];
  int y = aux1 - aux2 * dims[1];
  int t = aux2 / dims[2];
  int z = aux2 - t * dims[2];
  face_idx += (face_parity + t + z + y) & 1;

  int idx = face_idx;
  int gap, aux;

  // face_idx runs from 0 to (2*nFace*face_volume)-1
  switch (dim) {
    case 0: 
      // face_idx runs from 0 to nFace*X2*X3*X4-1
      gap = x[0] - nLayers;
      aux = face_idx;
      idx += face_num*gap + aux*(x[0]-1); 
      // idx runs from 0 to nFace*X1*X2*X3*X4-1 in jumps of X1
      idx += (idx/V)*(1-V);    
      break;
    case 1:
      // face_idx runs from 0 to nFace*X1*X3*X4-1
      gap = x[1] - nLayers;
      aux = face_idx / face_X;
      idx += face_num * gap* face_X + aux*(x[1]-1)*face_X;
      // idx = x1 + (face_idx/X1)*X1*X2
      // idx runs from x1 + X1...
      idx += (idx/V)*(x[0]-V);
      break;
    case 2:
      // face_idx runs from 0 to nFace*X1*X2*X4-1
      gap = x[2] - nLayers;
      aux = face_idx / face_XY;    
      idx += face_num * gap * face_XY +aux*(x[2]-1)*face_XY;
      // idx = x1 + x2*X1 + (face_idx/(X1*X2))*X1*X2*X3
      // idx runs over the volume in jumps of X1*X2*X3
      idx += (idx/V)*(x[1]*x[0]-V);
      break;
    case 3:
      // face_idx runs from 0 to nFace*X1*X2*X3-1
      gap = x[3] - nLayers;
      idx += face_num * gap * face_XYZ;
      // idx = x1 + x2*X1 + x3*X1*X2
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}


// compute full coordinates from an index into the face (used by the exterior Dslash kernels)
  template <int nLayers, typename Int>
static inline __device__ void coordsFromFaceIndex(int &idx, int &cb_idx, Int &X, Int &Y, Int &Z, Int &T, int face_idx,
    const int &face_volume, const int &dim, const int &face_num, const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3;
  int face_parity;
  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // compute coordinates from (checkerboard) face index

  face_idx *= 2;

  int x, y, z, t;

  if (!(face_X & 1)) { // face_X even
    //   t = face_idx / face_XYZ;
    //   z = (face_idx / face_XY) % face_Z;
    //   y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + t + z + y) & 1;
    //   x = face_idx % face_X;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    x = face_idx - aux1 * face_X;
    int aux2 = aux1 / face_Y;
    y = aux1 - aux2 * face_Y;
    t = aux2 / face_Z;
    z = aux2 - t * face_Z;
    x += (face_parity + t + z + y) & 1;
    // face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    t = face_idx / face_XYZ;
    z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + t + z) & 1;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else if (!(face_Z & 1)) { // face_Z even
    t = face_idx / face_XYZ;
    face_idx += (face_parity + t) & 1;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else {
    face_idx += face_parity;
    t = face_idx / face_XYZ; 
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  }

  //printf("Local sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);

  // need to convert to global coords, not face coords
  switch(dim) {
    case 0:
      x += face_num * (X1-nLayers);
      break;
    case 1:
      y += face_num * (X2-nLayers);
      break;
    case 2:
      z += face_num * (X3-nLayers);
      break;
    case 3:
      t += face_num * (X4-nLayers);
      break;
  }

  // compute index into the full local volume

  idx = X1*(X2*(X3*t + z) + y) + x; 

  // compute index into the checkerboard

  cb_idx = idx >> 1;

  X = x;
  Y = y;
  Z = z;
  T = t;  

  //printf("Global sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);
}


// compute coordinates from index into the checkerboard (used by the interior Dslash kernels)
  template <typename Int>
static __device__ __forceinline__ void coordsFromIndex(int &idx, Int &X, Int &Y, Int &Z, Int &T, const int &cb_idx, const int &parity)
{

  int &LX = X1;
  int &LY = X2;
  int &LZ = X3;
  int &XYZ = X3X2X1;
  int &XY = X2X1;

  idx = 2*cb_idx;

  int x, y, z, t;

  // The full field index is 
  // idx = x + y*X + z*X*Y + t*X*Y*Z
  // The parity of lattice site (x,y,z,t) 
  // is defined to be (x+y+z+t) & 1
  // 0 => even parity 
  // 1 => odd parity
  // cb_idx runs over the half volume
  // cb_idx = iidx/2 = (x + y*X + z*X*Y + t*X*Y*Z)/2
  //
  // We need to obtain idx from cb_idx + parity.
  // 
  // 1)  First, consider the case where X is even.
  // Then, y*X + z*X*Y + t*X*Y*Z is even and
  // 2*cb_idx = 2*(x/2) + y*X + z*X*Y + t*X*Y*Z
  // Since, 2*(x/2) is even, if y+z+t is even
  // (2*(x/2),y,z,t) is an even parity site.
  // Similarly, if y+z+t is odd
  // (2*(x/2),y,z,t) is an odd parity site. 
  // 
  // Note that (1+y+z+t)&1 = 1 for y+z+t even
  //      and  (1+y+z+t)&1 = 0 for y+z+t odd
  // Therefore, 
  // (2*/(x/2) + (1+y+z+t)&1, y, z, t) is odd.
  //
  // 2)  Consider the case where X is odd but Y is even.
  // Calculate 2*cb_idx
  // t = 2*cb_idx/XYZ
  // z = (2*cb_idx/XY) % Z
  //
  // Now, we  need to compute (x,y) for different parities.
  // To select a site with even parity, consider (z+t).
  // If (z+t) is even, this implies that (x+y) must also 
  // be even in order that (x+y+z+t) is even. 
  // Therefore,  x + y*X is even.
  // Thus, 2*cb_idx = idx 
  // and y =  (2*cb_idx/X) % Y
  // and x =  (2*cb_idx) % X;
  // 
  // On the other hand, if (z+t) is odd, (x+y) must be 
  // also be odd in order to get overall even parity. 
  // Then x + y*X is odd (since X is odd and either x or y is odd)
  // and 2*cb_idx = 2*(idx/2) = idx-1 =  x + y*X -1 + z*X*Y + t*X*Y*Z
  // => idx = 2*cb_idx + 1
  // and y = ((2*cb_idx + 1)/X) % Y
  // and x = (2*cb_idx + 1) % X
  //
  // To select a site with odd parity if (z+t) is even,
  // (x+y) must be odd, which, following the discussion above, implies that
  // y = ((2*cb_idx + 1)/X) % Y
  // x = (2*cb_idx + 1) % X
  // Finally, if (z+t) is odd (x+y) must be even to get overall odd parity, 
  // and 
  // y = ((2*cb_idx)/X) % Y
  // x = (2*cb_idx) % X
  // 
  // The code below covers these cases 
  // as well as the cases where X, Y are odd and Z is even,
  // and X,Y,Z are all odd



  if (!(LX & 1)) { // X even
    //   t = idx / XYZ;
    //   z = (idx / XY) % Z;
    //   y = (idx / X) % Y;
    //   idx += (parity + t + z + y) & 1;
    //   x = idx % X;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = idx / LX;
    x = idx - aux1 * LX;
    int aux2 = aux1 / LY;
    y = aux1 - aux2 * LY;
    t = aux2 / LZ;
    z = aux2 - t * LZ;
    aux1 = (parity + t + z + y) & 1;
    x += aux1;
    idx += aux1;
  } else if (!(LY & 1)) { // Y even
    t = idx / XYZ;
    z = (idx / XY) % LZ;
    idx += (parity + t + z) & 1;
    y = (idx / LX) % LY;
    x = idx % LX;
  } else if (!(LZ & 1)) { // Z even
    t = idx / XYZ;
    idx += (parity + t) & 1;
    z = (idx / XY) % LZ;
    y = (idx / LX) % LY;
    x = idx % LX;
  } else {
    idx += parity;
    t = idx / XYZ;
    z = (idx / XY) % LZ;
    y = (idx / LX) % LY;
    x = idx % LX;
  }

  X = x;
  Y = y;
  Z = z;
  T = t;
}

//Used in DW kernels only:

  template <int dim, int nLayers>
static inline __device__ int indexFromDWFaceIndex(int face_idx, const int &face_volume,
    const int &face_num, const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  //A.S.: Also used for computing offsets in physical lattice
  //A.S.: note that in the case of DW fermions one is dealing with 4d faces

  // intrinsic parity of the face depends on offset of first element, used for MPI DW as well
  int face_X = X1, face_Y = X2, face_Z = X3, face_T = X4;
  int face_parity;  

  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }

  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ = face_X * face_Y * face_Z;
  int face_XY = face_X * face_Y;

  // reconstruct full face index from index into the checkerboard

  face_idx *= 2;

  if (!(face_X & 1)) { // face_X even
    //   int s = face_idx / face_XYZT;    
    //   int t = (face_idx / face_XYZ) % face_T;
    //   int z = (face_idx / face_XY) % face_Z;
    //   int y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + s + t + z + y) & 1;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    int aux2 = aux1 / face_Y;
    int aux3 = aux2 / face_Z;
    int y = aux1 - aux2 * face_Y;
    int z = aux2 - aux3 * face_Z;    
    int s = aux3 / face_T;
    int t = aux3 - s * face_T;
    face_idx += (face_parity + s + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    int s = face_idx / face_XYZT;    
    int t = (face_idx / face_XYZ) % face_T;
    int z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + s + t + z) & 1;
  } else if (!(face_Z & 1)) { // face_Z even
    int s = face_idx / face_XYZT;        
    int t = (face_idx / face_XYZ) % face_T;
    face_idx += (face_parity + s + t) & 1;
  } else if(!(face_T)){
    int s = face_idx / face_XYZT;        
    face_idx += (face_parity + s) & 1;
  }else{    
    face_idx += face_parity;
  }

  // compute index into the full local volume

  int idx = face_idx;
  int gap, aux;

  switch (dim) {
    case 0:
      gap = X1 - nLayers;
      aux = face_idx / face_X;
      idx += (aux + face_num) * gap;
      break;
    case 1:
      gap = X2 - nLayers;
      aux = face_idx / face_XY;
      idx += (aux + face_num) * gap * face_X;
      break;
    case 2:
      gap = X3 - nLayers;
      aux = face_idx / face_XYZ;
      idx += (aux + face_num) * gap * face_XY;
      break;
    case 3:
      gap = X4 - nLayers;
      aux = face_idx / face_XYZT;
      idx += (aux + face_num) * gap * face_XYZ;
      break;
  }

  // return index into the checkerboard

  return idx >> 1;
}


// compute full coordinates from an index into the face (used by the exterior Dslash kernels)
  template <int nLayers, typename Int>
static inline __device__ void coordsFromDWFaceIndex(int &cb_idx, Int &X, Int &Y, Int &Z, Int &T, Int &S, int face_idx,
    const int &face_volume, const int &dim, const int &face_num, const int &parity)
{
  // dimensions of the face (FIXME: optimize using constant cache)

  int face_X = X1, face_Y = X2, face_Z = X3, face_T = X4;
  int face_parity;
  switch (dim) {
    case 0:
      face_X = nLayers;
      face_parity = (parity + face_num * (X1 - nLayers)) & 1;
      break;
    case 1:
      face_Y = nLayers;
      face_parity = (parity + face_num * (X2 - nLayers)) & 1;
      break;
    case 2:
      face_Z = nLayers;
      face_parity = (parity + face_num * (X3 - nLayers)) & 1;
      break;
    case 3:
      face_T = nLayers;    
      face_parity = (parity + face_num * (X4 - nLayers)) & 1;
      break;
  }
  int face_XYZT = face_X * face_Y * face_Z * face_T;  
  int face_XYZ  = face_X * face_Y * face_Z;
  int face_XY   = face_X * face_Y;

  // compute coordinates from (checkerboard) face index

  face_idx *= 2;

  int x, y, z, t, s;

  if (!(face_X & 1)) { // face_X even
    //   s = face_idx / face_XYZT;        
    //   t = (face_idx / face_XYZ) % face_T;
    //   z = (face_idx / face_XY) % face_Z;
    //   y = (face_idx / face_X) % face_Y;
    //   face_idx += (face_parity + s + t + z + y) & 1;
    //   x = face_idx % face_X;
    // equivalent to the above, but with fewer divisions/mods:
    int aux1 = face_idx / face_X;
    x = face_idx - aux1 * face_X;
    int aux2 = aux1 / face_Y;
    y = aux1 - aux2 * face_Y;
    int aux3 = aux2 / face_Z;
    z = aux2 - aux3 * face_Z;
    s = aux3 / face_T;
    t = aux3 - s * face_T;
    x += (face_parity + s + t + z + y) & 1;
    // face_idx += (face_parity + t + z + y) & 1;
  } else if (!(face_Y & 1)) { // face_Y even
    s = face_idx / face_XYZT;    
    t = (face_idx / face_XYZ) % face_T;
    z = (face_idx / face_XY) % face_Z;
    face_idx += (face_parity + s + t + z) & 1;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else if (!(face_Z & 1)) { // face_Z even
    s = face_idx / face_XYZT;    
    t = (face_idx / face_XYZ) % face_T;
    face_idx += (face_parity + s + t) & 1;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  } else {
    s = face_idx / face_XYZT;        
    face_idx += face_parity;
    t = (face_idx / face_XYZ) % face_T;
    z = (face_idx / face_XY) % face_Z;
    y = (face_idx / face_X) % face_Y;
    x = face_idx % face_X;
  }

  //printf("Local sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);

  // need to convert to global coords, not face coords
  switch(dim) {
    case 0:
      x += face_num * (X1-nLayers);
      break;
    case 1:
      y += face_num * (X2-nLayers);
      break;
    case 2:
      z += face_num * (X3-nLayers);
      break;
    case 3:
      t += face_num * (X4-nLayers);
      break;
  }

  // compute index into the checkerboard

  cb_idx = (X1*(X2*(X3*(X4*s + t) + z) + y) + x) >> 1;

  X = x;
  Y = y;
  Z = z;
  T = t;  
  S = s;
  //printf("Global sid %d (%d, %d, %d, %d)\n", cb_int, x, y, z, t);
}


// routines for packing the ghost zones (multi-GPU only)

#ifdef MULTI_GPU

#if defined(GPU_WILSON_DIRAC) || defined(GPU_DOMAIN_WALL_DIRAC)

// double precision
#if (defined DIRECT_ACCESS_WILSON_PACK_SPINOR) || (defined FERMI_NO_DBLE_TEX)
#define READ_SPINOR READ_SPINOR_DOUBLE
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_DOUBLE_TEX
#define READ_SPINOR_UP READ_SPINOR_DOUBLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_DOUBLE_DOWN_TEX
#define SPINORTEX spinorTexDouble
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_DOUBLE2
#define SPINOR_DOUBLE
  template <int dim, int dagger>
static inline __device__ void packFaceWilsonCore(double2 *out, float *outNorm, const double2 *in, const float *inNorm,
    const int &idx, const int &face_idx, const int &face_volume, const int &face_num)
{
#if (__COMPUTE_CAPABILITY__ >= 130)
  if (dagger) {
#include "wilson_pack_face_dagger_core.h"
  } else {
#include "wilson_pack_face_core.h"
  }
#endif // (__COMPUTE_CAPABILITY__ >= 130)
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR
#undef SPINOR_DOUBLE


// single precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_SINGLE
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_SINGLE_TEX
#define READ_SPINOR_UP READ_SPINOR_SINGLE_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_SINGLE_DOWN_TEX
#define SPINORTEX spinorTexSingle
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_FLOAT4
  template <int dim, int dagger>
static inline __device__ void packFaceWilsonCore(float4 *out, float *outNorm, const float4 *in, const float *inNorm,
    const int &idx, const int &face_idx, const int &face_volume, const int &face_num)
{
  if (dagger) {
#include "wilson_pack_face_dagger_core.h"
  } else {
#include "wilson_pack_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR


// half precision
#ifdef DIRECT_ACCESS_WILSON_PACK_SPINOR
#define READ_SPINOR READ_SPINOR_HALF
#define READ_SPINOR_UP READ_SPINOR_HALF_UP
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN
#define SPINORTEX in
#else
#define READ_SPINOR READ_SPINOR_HALF_TEX
#define READ_SPINOR_UP READ_SPINOR_HALF_UP_TEX
#define READ_SPINOR_DOWN READ_SPINOR_HALF_DOWN_TEX
#define SPINORTEX spinorTexHalf
#endif
#define WRITE_HALF_SPINOR WRITE_HALF_SPINOR_SHORT4
  template <int dim, int dagger>
static inline __device__ void packFaceWilsonCore(short4 *out, float *outNorm, const short4 *in, const float *inNorm,
    const int &idx, const int &face_idx, const int &face_volume, const int &face_num)
{
  if (dagger) {
#include "wilson_pack_face_dagger_core.h"
  } else {
#include "wilson_pack_face_core.h"
  }
}
#undef READ_SPINOR
#undef READ_SPINOR_UP
#undef READ_SPINOR_DOWN
#undef SPINORTEX
#undef WRITE_HALF_SPINOR


  template <int dim, int dagger, typename FloatN>
__global__ void packFaceWilsonKernel(FloatN *out, float *outNorm, const FloatN *in, 
    const float *inNorm, const int parity)
{
  const int nFace = 1; // 1 face for Wilson
  const int Nint = 12; // output is spin projected
  size_t faceBytes = nFace*ghostFace[dim]*Nint*sizeof(out->x);
  if (sizeof(FloatN)==sizeof(short4)) faceBytes += nFace*ghostFace[dim]*sizeof(float);

  int face_volume = ghostFace[dim];
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (face_idx >= 2*nFace*face_volume) return;

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;

  // compute an index into the local volume from the index into the face
  const int idx = indexFromFaceIndex<dim, nFace>(face_idx, face_volume, face_num, parity);

  if (face_num) {
    out = (FloatN*)((char*)out + faceBytes);
    outNorm = (float*)((char*)outNorm + faceBytes);
  }

  // read spinor, spin-project, and write half spinor to face
  packFaceWilsonCore<dim, dagger>(out, outNorm, in, inNorm, idx, face_idx, face_volume, face_num);
}

#endif // GPU_WILSON_DIRAC || GPU_DOMAIN_WALL_DIRAC


template <typename FloatN>
class PackFaceWilson : public Tunable {

  private:
    FloatN *faces;
    float *facesNorm;
    const FloatN *in;
    const float *inNorm;
    const int dim;
    const int dagger;
    const int parity;
    const int *X;
    const int *ghostFace;
    const int stride;
    const int nFace;

    int sharedBytesPerThread() const { return 0; }
    int sharedBytesPerBlock(const TuneParam &param) const { return 0; }

    bool advanceGridDim(TuneParam &param) const { return false; } // Don't tune the grid dimensions.
    bool advanceBlockDim(TuneParam &param) const {
      bool advance = Tunable::advanceBlockDim(param);
      unsigned int threads = ghostFace[dim]*nFace*2; // 2 for forwards and backwards faces
      if (advance) param.grid = dim3( (threads+param.block.x-1) / param.block.x, 1, 1);
      return advance;
    }

  public:
    PackFaceWilson(FloatN *faces, float *facesNorm, const FloatN *in, const float *inNorm, 
        const int dim, const int dagger, const int parity, const int *X, 
        const int *ghostFace, const int stride)
      : faces(faces), facesNorm(facesNorm), in(in), inNorm(inNorm), dim(dim), dagger(dagger), 
      parity(parity), X(X), ghostFace(ghostFace), stride(stride), nFace(1) { }
    virtual ~PackFaceWilson() { }

    TuneKey tuneKey() const {
      std::stringstream vol, aux;
      vol << X[0] << "x";
      vol << X[1] << "x";
      vol << X[2] << "x";
      vol << X[3];    
      aux << "dim=" << dim << ",stride=" << stride << ",prec=" << sizeof(((FloatN*)0)->x);
      return TuneKey(vol.str(), typeid(*this).name(), aux.str());
    }  

    void apply(const cudaStream_t &stream) {
      TuneParam tp = tuneLaunch(*this, dslashTuning, verbosity);

#ifdef GPU_WILSON_DIRAC
      if (dagger) {
        switch (dim) {
          case 0: packFaceWilsonKernel<0,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
          case 1: packFaceWilsonKernel<1,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
          case 2: packFaceWilsonKernel<2,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
          case 3: packFaceWilsonKernel<3,1><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
        }
      } else {
        switch (dim) {
          case 0: packFaceWilsonKernel<0,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
          case 1: packFaceWilsonKernel<1,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
          case 2: packFaceWilsonKernel<2,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
          case 3: packFaceWilsonKernel<3,0><<<tp.grid, tp.block, tp.shared_bytes, stream>>>(faces, facesNorm, in, inNorm, parity); break;
        }
      }
#else
      errorQuda("Wilson face packing kernel is not built");
#endif  
    }

    long long flops() const { return 12*ghostFace[dim]; }
    long long bytes() const { 
      int Nint = 36;
      size_t faceBytes = 2*nFace*ghostFace[dim]*Nint*sizeof(((FloatN*)0)->x);
      if (sizeof(((FloatN*)0)->x) == QUDA_HALF_PRECISION) 
        faceBytes += nFace*ghostFace[dim]*sizeof(float);
      return faceBytes;
    }

    virtual void initTuneParam(TuneParam &param) const
    {
      Tunable::initTuneParam(param);
      unsigned int threads = ghostFace[dim]*nFace*2; // 2 for forwards and backwards faces
      param.grid = dim3( (threads+param.block.x-1) / param.block.x, 1, 1);
    }

    /** sets default values for when tuning is disabled */
    virtual void defaultTuneParam(TuneParam &param) const
    {
      Tunable::defaultTuneParam(param);
      unsigned int threads = ghostFace[dim]*nFace*2; // 2 for forwards and backwards faces
      param.grid = dim3( (threads+param.block.x-1) / param.block.x, 1, 1);
    }

};

void packFaceWilson(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
    const int parity, const cudaStream_t &stream) {
  const int nFace = 1; // 1 face for Wilson

  unsigned int threads = in.GhostFace()[dim]*nFace*2;
  dim3 blockDim(64, 1, 1); // TODO: make this a parameter for auto-tuning
  dim3 gridDim( (threads+blockDim.x-1) / blockDim.x, 1, 1);

  // compute location of norm zone
  int Nint = in.Ncolor() * in.Nspin(); // assume spin projection
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision());

  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      {
        PackFaceWilson<double2> pack((double2*)ghost_buf, ghostNorm, (double2*)in.V(), (float*)in.Norm(), 
            dim, dagger, parity, in.X(), in.GhostFace(), in.Stride());
        pack.apply(stream);
      }
      break;
    case QUDA_SINGLE_PRECISION:
      {
        PackFaceWilson<float4> pack((float4*)ghost_buf, ghostNorm, (float4*)in.V(), (float*)in.Norm(), 
            dim, dagger, parity, in.X(), in.GhostFace(), in.Stride());
        pack.apply(stream);
      }
      break;
    case QUDA_HALF_PRECISION:
      {
        PackFaceWilson<short4> pack((short4*)ghost_buf, ghostNorm, (short4*)in.V(), (float*)in.Norm(), 
            dim, dagger, parity, in.X(), in.GhostFace(), in.Stride());
        pack.apply(stream);
      }
      break;
  }  
}

#ifdef GPU_STAGGERED_DIRAC

#if (defined DIRECT_ACCESS_PACK) || (defined FERMI_NO_DBLE_TEX)
template <typename Float2>
__device__ void packSpinor(Float2 *out, float *outNorm, int out_idx, int out_stride, 
    const Float2 *in, const float *inNorm, int in_idx, int in_stride) {
  __syncthreads(); 
  printf("in_idx = %d; out[%d, %d, %d] = (%lf, %lf, %lf, %lf, %lf, %lf)\n",in_idx, out_idx, out_idx+out_stride, out_idx+2*out_stride,
                                                              in[in_idx ].x, in[in_idx].y, 
                                                              in[in_idx + in_stride].x, in[in_idx + in_stride].y,
                                                              in[in_idx + 2*in_stride].x, in[in_idx + 2*in_stride].y);
                                                                
  __syncthreads();

  out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
}	
template<> __device__ void packSpinor(short2 *out, float *outNorm, int out_idx, int out_stride, 
    const short2 *in, const float *inNorm, int in_idx, int in_stride) {
  out[out_idx + 0*out_stride] = in[in_idx + 0*in_stride];
  out[out_idx + 1*out_stride] = in[in_idx + 1*in_stride];
  out[out_idx + 2*out_stride] = in[in_idx + 2*in_stride];
  outNorm[out_idx] = inNorm[in_idx];
}
#else

__device__ void packSpinor(double2 *out, float *outNorm, int out_idx, int out_stride, 
    const double2 *in, const float *inNorm, int in_idx, int in_stride) {
  out[out_idx + 0*out_stride] = fetch_double2(spinorTexDouble, in_idx + 0*in_stride);
  out[out_idx + 1*out_stride] = fetch_double2(spinorTexDouble, in_idx + 1*in_stride);
  out[out_idx + 2*out_stride] = fetch_double2(spinorTexDouble, in_idx + 2*in_stride);
}	
__device__ void packSpinor(float2 *out, float *outNorm, int out_idx, int out_stride, 
    const float2 *in, const float *inNorm, int in_idx, int in_stride) {
  out[out_idx + 0*out_stride] = tex1Dfetch(spinorTexSingle2, in_idx + 0*in_stride);
  out[out_idx + 1*out_stride] = tex1Dfetch(spinorTexSingle2, in_idx + 1*in_stride);
  out[out_idx + 2*out_stride] = tex1Dfetch(spinorTexSingle2, in_idx + 2*in_stride);	
}

// this is rather dumb: undoing the texture load because cudaNormalizedReadMode is used
// should really bind to an appropriate texture instead of reusing
static inline __device__ short2 float22short2(float c, float2 a) {
  return make_short2((short)(a.x*c*MAX_SHORT), (short)(a.y*c*MAX_SHORT));
}

__device__ void packSpinor(short2 *out, float *outNorm, int out_idx, int out_stride, 
    const short2 *in, const float *inNorm, int in_idx, int in_stride) {
  out[out_idx + 0*out_stride] = float22short2(1.0f, tex1Dfetch(spinorTexHalf2, in_idx + 0*in_stride));
  out[out_idx + 1*out_stride] = float22short2(1.0f, tex1Dfetch(spinorTexHalf2, in_idx + 1*in_stride));
  out[out_idx + 2*out_stride] = float22short2(1.0f, tex1Dfetch(spinorTexHalf2, in_idx + 2*in_stride));
  outNorm[out_idx] = tex1Dfetch(spinorTexHalfNorm, in_idx);
}
#endif

//
// TODO: add support for textured reads

  template <int dim, int ishalf, int nFace, typename Float2>
__global__ void packFaceStaggeredKernel(Float2 *out, float *outNorm, const Float2 *in, 
    const float *inNorm, const int parity, int x1, int x2, int x3, int x4)
{

  //  const int nFace = 3; //3 faces for asqtad
  const int Nint = 6; // number of internal degrees of freedom
  size_t faceBytes = nFace*ghostFace[dim]*Nint*sizeof(out->x); 
  if (ishalf) faceBytes += nFace*ghostFace[dim]*sizeof(float);

  int face_volume = ghostFace[dim]; // ghostFace[0] = X[1]*X[2]*X[3]/2, etc.
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(face_idx == 0){ 
    printf("In packFaceStaggered Kernel\n");
  }
  int x[4] = {x1, x2, x3, x4};

  if (face_idx >= 2*nFace*face_volume) return;

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;

  // compute an index into the local volume from the index into the face
  //const int idx = indexFromFaceIndexAsqtad<dim, nFace>(face_idx, face_volume, face_num, parity);
  int idx = indexFromFaceIndexAsqtad<dim, nFace>(face_idx, face_volume, face_num, parity, x);


  // Test to see if this works!!
  // Does coordsFromIndex work??
  //  int x, y, z, t, full_idx; 
  //  coordsFromIndex(full_idx , x, y, z, t, idx, parity);

  //  idx = X1*X2*X3*t + X1*X2*z + X1*y + x;
  //  idx = idx >> 1;

  if(face_idx == 0) printf("In packFaceStaggeredKernel: About to call packSpinor\n");

  if (face_num) {
    out = (Float2*)((char*)out + faceBytes);
    outNorm = (float*)((char*)outNorm + faceBytes);
  }

/*
  if(face_idx == 0 && face_num == 0) {
    printf("out_stride = %d\n", nFace*face_volume);
    printf("in_stride = %d\n", sp_stride); 
    printf("in_idx = %d; out = (%lf, %lf, %lf, %lf, %lf, %lf)\n",idx, 
                                                              in[idx ].x, in[idx].y, 
                                                              in[idx + sp_stride].x, in[idx + sp_stride].y,
                                                              in[idx + 2*sp_stride].x, in[idx + 2*sp_stride].y);
                                                                
  }
*/ 
  packSpinor(out, outNorm, face_idx, nFace*face_volume, in, inNorm, idx, sp_stride);


  if(face_idx == 0) printf("Call to packStaggeredKernel complete\n");
  /*  Float2 tmp1 = in[idx + 0*sp_stride];
      Float2 tmp2 = in[idx + 1*sp_stride];
      Float2 tmp3 = in[idx + 2*sp_stride];

      out[face_idx + 0*nFace*face_volume] = tmp1;
      out[face_idx + 1*nFace*face_volume] = tmp2;
      out[face_idx + 2*nFace*face_volume] = tmp3;

      if (ishalf) outNorm[face_idx] = inNorm[idx];*/
}

  template <int dim, int ishalf, int nFace, typename Float2>
__global__ void unpackFaceStaggeredKernel(Float2 *out, 
    float *outNorm,
    Float2* in,
    float* inNorm,
    const int parity)
//	const int bx, // (bx, by, bz, bt) is the border region for the overlapped domains
//  const int by,
//  const int bz, 
//		const int bt)
{
  const int Nint = 6;
  size_t faceBytes = nFace*ghostFace[dim]*Nint*sizeof(out->x);
  if(ishalf) faceBytes += nFace*ghostFace[dim]*sizeof(float);

  int face_volume = ghostFace[dim];
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if(face_idx >= 2*nFace*face_volume) return;

  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;
  /* 
  // compute an index into the local volume from the index into the face 
  const int cb_idx = indexFromFaceIndexAsqtad<dim,nFace>(face_idx, face_volume, face_num, parity);
  int idx = 2*cb_idx;
  // compute an index into the larger volume 
  //  int x = idx % X1;
  //  int y = (idx/X1) % X2;
  //  int z = (idx/X2*X1) % X3;
  //  int t = idx/(X1*X2*X3);
  //  int even_odd;
  // Work out the parity of the lattice site and add to zero

  // Not sure if packSpinor will work out of the box, since it may use textures.
  // packSpinor(out, outNorm, idx, sp_stride, (Float2*)(((char*)in+face_num*faceBytes)), (float*)((char*)inNorm+face_num*faceBytes), face_idx, nFace*face_volume);

  Float2* tmp = (Float2*)(((char*)in + face_num*faceBytes));
  float* tmpNorm = (float*)((char*)inNorm + face_num*faceBytes);

  out[cb_idx] = tmp[face_idx];
  out[cb_idx +   sp_stride]  = tmp[face_idx +   nFace*face_volume];
  out[cb_idx + 2*sp_stride]  = tmp[face_idx + 2*nFace*face_volume];
  if(ishalf) outNorm[idx] = tmpNorm[face_idx];
   */
  return;
}



#endif // GPU_STAGGERED_DIRAC

template<typename T>
struct isShort2
{
  static const int val = 0;
};

template<>
struct isShort2<short2>
{
  static const int val = 1;
};

template<typename T> const int isShort2<T>::val;


  template <int nFace, typename Real2>
void packFaceStaggeredKernelWrapper(Real2 *faces, float *facesNorm, Real2 *in, float *inNorm, int dim,
    const int parity, const dim3 &gridDim, const dim3 &blockDim, 
    const cudaStream_t &stream, int* y)
{
#ifdef GPU_STAGGERED_DIRAC
  const int ishalf = isShort2<Real2>::val;
  switch (dim) {
    case 0: packFaceStaggeredKernel<0,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity, 
                y[0], y[1], y[2], y[3]); break;
    case 1: packFaceStaggeredKernel<1,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity,
                y[0], y[1], y[2], y[3]); break;
    case 2: packFaceStaggeredKernel<2,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity, 
                y[0], y[1], y[2], y[3]); break;
    case 3: packFaceStaggeredKernel<3,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity, 
                y[0], y[1], y[2], y[3]); break;
  }
#else
  errorQuda("Staggered face packing kernel is not built");
#endif  
}



  template<int nFace, typename Real2>
void unpackFaceStaggeredKernelWrapper(Real2 *faces, float *facesNorm, Real2 *in, float *inNorm, int dim,
    const int parity, const dim3 &gridDim, const dim3 &blockDim,
    const cudaStream_t &stream)
{
#ifdef GPU_STAGGERED_DIRAC
  const int ishalf = isShort2<Real2>::val;
  switch(dim) {
    case 0: unpackFaceStaggeredKernel<0,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
    case 1: unpackFaceStaggeredKernel<1,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break; 
    case 2: unpackFaceStaggeredKernel<2,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
    case 3: unpackFaceStaggeredKernel<3,ishalf,nFace><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
  }
#else
  errorQuda("Staggered face unpacking kernel is not built");
#endif
}


template<int nFace>
void unpackFaceStaggered(cudaColorSpinorField &in, void* ghost_buf,  const int dim, const int dagger,
    const int parity, const cudaStream_t &stream){

  unsigned int threads = in.GhostFace()[dim]*nFace*2;
  dim3 blockDim(64, 1, 1);
  dim3 gridDim( (threads+blockDim.x-1)/blockDim.x, 1, 1);

  // compute location of norm zone
  int Nint = 6;
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision());

  switch(in.Precision()){
    case QUDA_DOUBLE_PRECISION:
      unpackFaceStaggeredKernelWrapper<nFace>((double2*)in.V(), (float*)in.Norm(),(double2*)ghost_buf, ghostNorm, dim, parity, gridDim, blockDim, stream);
      break;
    case QUDA_SINGLE_PRECISION:
      unpackFaceStaggeredKernelWrapper<nFace>((float2*)in.V(), (float*)in.Norm(), (float2*)ghost_buf, ghostNorm, dim, parity, gridDim, blockDim, stream);
      break;  
    case  QUDA_HALF_PRECISION:
      unpackFaceStaggeredKernelWrapper<nFace>((short2*)in.V(), (float*)in.Norm(), (short2*)ghost_buf, ghostNorm, dim, parity, gridDim, blockDim, stream);
      break;
  } 
  return;
}




template<int nFace>
void packFaceStaggered(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
    const int parity, const cudaStream_t &stream, int* y) {
  //  const int nFace = 3; //3 faces for asqtad

  unsigned int threads = in.GhostFace()[dim]*nFace*2; 
  dim3 blockDim(64, 1, 1); // TODO: make this a parameter for auto-tuning
  dim3 gridDim( (threads+blockDim.x-1) / blockDim.x, 1, 1);

  // compute location of norm zone
  int Nint = 6;
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision());

  printfQuda("packFaceStaggered: input address = %p\n", in.V());
  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      packFaceStaggeredKernelWrapper<nFace>((double2*)ghost_buf, ghostNorm, (double2*)in.V(), (float*)in.Norm(), 
          dim, parity, gridDim, blockDim, stream, y);
      break;
    case QUDA_SINGLE_PRECISION:
      packFaceStaggeredKernelWrapper<nFace>((float2*)ghost_buf, ghostNorm, (float2*)in.V(), (float*)in.Norm(), 
          dim, parity, gridDim, blockDim, stream, y);
      break;
    case QUDA_HALF_PRECISION:
      packFaceStaggeredKernelWrapper<nFace>((short2*)ghost_buf, ghostNorm, (short2*)in.V(), (float*)in.Norm(), 
          dim, parity, gridDim, blockDim, stream, y);
      break;
  }  
}

void packFace(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
    const int parity, const cudaStream_t &stream, int* y)
{

  if(in.Nspin() == 1){
    printfQuda("Calling packFaceStaggered\n");
    printfQuda("y = %d %d %d %d\n", y[0], y[1], y[2], y[3]);
    switch(in.Nface()){
      case 1: packFaceStaggered<1>(ghost_buf, in, dim, dagger, parity, stream, y); break;
      case 2: packFaceStaggered<2>(ghost_buf, in, dim, dagger, parity, stream, y); break;
      case 3: packFaceStaggered<3>(ghost_buf, in, dim, dagger, parity, stream, y); break;
      case 4: packFaceStaggered<4>(ghost_buf, in, dim, dagger, parity, stream, y); break;
      default: errorQuda("Only nFace 1/2/3/4 supported for staggered fermions\n"); break;
    }
  }
}

void packFace(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger, 
    const int parity, const cudaStream_t &stream){
  packFaceWilson(ghost_buf, in, dim, dagger, parity, stream);
}


void unpackFace(cudaColorSpinorField &out, void* ghost_buf, const int dim, const int dagger, 
    const int parity, const cudaStream_t &stream)
{
  if(out.Nspin() == 1){
    switch(out.Nface()){
      case 2: unpackFaceStaggered<2>(out, ghost_buf, dim, dagger, parity, stream); break;
      case 4: unpackFaceStaggered<4>(out, ghost_buf, dim, dagger, parity, stream); break;
      default: errorQuda("Only border width 2/4 supported for staggered fermions\n"); break;
    }
  }else{
    errorQuda("Unpacking only supported for staggered fermions\n");
  }
  return;
}

//BEGIN NEW
#ifdef GPU_DOMAIN_WALL_DIRAC
  template <int dim, int dagger, typename FloatN>
__global__ void packFaceDWKernel(FloatN *out, float *outNorm, const FloatN *in, const float *inNorm, const int parity)
{
  const int nFace = 1; // 1 face for Wilson
  const int Nint = 12; // output is spin projected
  size_t faceBytes = nFace*Ls*ghostFace[dim]*Nint*sizeof(out->x);
  if (sizeof(FloatN)==sizeof(short4)) faceBytes += nFace*Ls*ghostFace[dim]*sizeof(float);

  int face_volume = Ls*ghostFace[dim];
  int face_idx = blockIdx.x*blockDim.x + threadIdx.x;

  if (face_idx >= 2*nFace*face_volume) return;

  // face_num determines which end of the lattice we are packing: 0 = beginning, 1 = end
  const int face_num = (face_idx >= nFace*face_volume) ? 1 : 0;
  face_idx -= face_num*nFace*face_volume;

  // compute an index into the local volume from the index into the face
  int idx = indexFromDWFaceIndex<dim, 1>(face_idx, face_volume, face_num, parity);

  if (face_num) {
    out = (FloatN*)((char*)out + faceBytes);
    outNorm = (float*)((char*)outNorm + faceBytes);
  }

  // read spinor, spin-project, and write half spinor to face (the same kernel as for Wilson): 
  packFaceWilsonCore<dim, dagger>(out, outNorm, in, inNorm, idx, face_idx, face_volume, face_num);
}
#endif

template <typename FloatN>
void packFaceDW(FloatN *faces, float *facesNorm, const FloatN *in, const float *inNorm, 
    const int dim, const int dagger, const int parity, 
    const dim3 &gridDim, const dim3 &blockDim, const cudaStream_t &stream){
#ifdef GPU_DOMAIN_WALL_DIRAC  
  if (dagger) {
    switch (dim) {
      case 0: packFaceDWKernel<0,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
      case 1: packFaceDWKernel<1,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
      case 2: packFaceDWKernel<2,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
      case 3: packFaceDWKernel<3,1><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
    }
  } else {
    switch (dim) {
      case 0: packFaceDWKernel<0,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
      case 1: packFaceDWKernel<1,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
      case 2: packFaceDWKernel<2,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
      case 3: packFaceDWKernel<3,0><<<gridDim, blockDim, 0, stream>>>(faces, facesNorm, in, inNorm, parity); break;
    }
  }
#else
  errorQuda("DW face packing kernel is not built");
#endif  

}

void packFaceDW(void *ghost_buf, cudaColorSpinorField &in, const int dim, const int dagger,  const int parity, const cudaStream_t &stream) {
#ifdef GPU_WILSON_DIRAC
  const int nFace = 1; // 1 face for Wilson
  dim3 blockDim(64, 1, 1); // TODO: make this a parameter for auto-tuning
  dim3 gridDim( (2*in.GhostFace()[dim]+blockDim.x-1) / blockDim.x, 1, 1);

  int Nint = in.Ncolor() * in.Nspin(); // assume spin projection
  float *ghostNorm = (float*)((char*)ghost_buf + Nint*nFace*in.GhostFace()[dim]*in.Precision()); // norm zone

  //printfQuda("Starting face packing: dimension = %d, direction = %d, face size = %d\n", dim, dir, in.ghostFace[dim]);
  switch(in.Precision()) {
    case QUDA_DOUBLE_PRECISION:
      packFaceDW((double2*)ghost_buf, ghostNorm, (double2*)in.V(), (float*)in.Norm(), 
          dim, dagger, parity, gridDim, blockDim, stream);
      break;
    case QUDA_SINGLE_PRECISION:
      packFaceDW((float4*)ghost_buf, ghostNorm, (float4*)in.V(), (float*)in.Norm(), 
          dim, dagger, parity, gridDim, blockDim, stream);
      break;
    case QUDA_HALF_PRECISION:
      packFaceDW((short4*)ghost_buf, ghostNorm, (short4*)in.V(), (float*)in.Norm(), 
          dim, dagger, parity, gridDim, blockDim, stream);
      break;
  }  
  //printfQuda("Completed face packing\n", dim, dir, ghostFace[dir]);
#else
  errorQuda("DW face parking routines are not built. Check that both GPU_WILSON_DIRAC and GPU_DOMAIN_WALL_DIRAC compiler flags are set.");
#endif //GPU_WILSON_DIRAC
}
//END NEW

#endif // MULTI_GPU


