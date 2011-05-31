//

#ifndef gpu_impl_h__
#define gpu_impl_h__ 1

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

// masks and class codes.  note that each implementation class may have unique restrictions
#define GPU_IMPL_CLASS_MASK  0xff00
#define GPU_IMPL_PARAM_MASK  0x00ff
#define GPU_IMPL_CLASS_NONE  0x0000
#define GPU_IMPL_CLASS_Y1    0x0100 // (no specfic restrictions on ny)
#define GPU_IMPL_CLASS_Y2    0x0200 // ny must be multiple of 2
#define GPU_IMPL_CLASS_YN    0x0300 // (no specfic restrictions on ny)
#define GPU_IMPL_CLASS_Y1Z   0x0500 // nz must be multiple of GPU_IMPL_PARAM.  loop is unrolled GPU_IMPL_PARAM times.  identical to GPU_IMPL_CLASS_Y1 if GPU_IMPL_PARAM == 1.
#define GPU_IMPL_CLASS_Y2Z   0x0400 // ny must be multiple of 2.  nz must be multiple of GPU_IMPL_PARAM.  loop is unrolled GPU_IMPL_PARAM times.  identical to GPU_IMPL_CLASS_Y2 if GPU_IMPL_PARAM == 1.

// The possibilities for 'GPU_IMPL'.
#define GPU_IMPL_NONE  0
#define GPU_IMPL_Y1    (GPU_IMPL_CLASS_Y1)
#define GPU_IMPL_Y2    (GPU_IMPL_CLASS_Y2)
#define GPU_IMPL_YN_1  (GPU_IMPL_CLASS_YN | 1)
#define GPU_IMPL_YN_2  (GPU_IMPL_CLASS_YN | 2)
#define GPU_IMPL_YN_4  (GPU_IMPL_CLASS_YN | 4)
#define GPU_IMPL_YN_8  (GPU_IMPL_CLASS_YN | 8)
#define GPU_IMPL_YN_16 (GPU_IMPL_CLASS_YN | 16)
#define GPU_IMPL_Y1Z1  (GPU_IMPL_CLASS_Y1Z | 1)
#define GPU_IMPL_Y1Z2  (GPU_IMPL_CLASS_Y1Z | 2)
#define GPU_IMPL_Y1Z4  (GPU_IMPL_CLASS_Y1Z | 4)
#define GPU_IMPL_Y1Z8  (GPU_IMPL_CLASS_Y1Z | 8)
#define GPU_IMPL_Y1Z16 (GPU_IMPL_CLASS_Y1Z | 16)
#define GPU_IMPL_Y1Z32 (GPU_IMPL_CLASS_Y1Z | 32)
#define GPU_IMPL_Y2Z1  (GPU_IMPL_CLASS_Y2Z | 1)
#define GPU_IMPL_Y2Z2  (GPU_IMPL_CLASS_Y2Z | 2)
#define GPU_IMPL_Y2Z4  (GPU_IMPL_CLASS_Y2Z | 4)
#define GPU_IMPL_Y2Z8  (GPU_IMPL_CLASS_Y2Z | 8)
#define GPU_IMPL_Y2Z16 (GPU_IMPL_CLASS_Y2Z | 16)
#define GPU_IMPL_Y2Z32 (GPU_IMPL_CLASS_Y2Z | 32)

/* Implementation selector.  Change this macro to select best
 * implementation.  You may specifiy it here using symbols defined
 * above, or specifiy it at different place using hex value (e.g. on
 * invoking nvcc, -DGPU_IMPL=0x0408 for GPU_IMPL_Y2Z8).  */
#ifndef GPU_IMPL
#define GPU_IMPL GPU_IMPL_Y2Z8
#endif

#define GPU_IMPL_CLASS (GPU_IMPL & GPU_IMPL_CLASS_MASK)
#define GPU_IMPL_PARAM (GPU_IMPL & GPU_IMPL_PARAM_MASK)

#if GPU_IMPL == GPU_IMPL_NONE || GPU_IMPL_CLASS == GPU_IMPL_CLASS_NONE
// NOTE: using the fact that misspelled macros are treated as 0
#error invalid GPU_IMPL selected (misspelled?)
#endif

template <typename Float, bool checkBottom>
__device__ void gpu_impl_sub(Float const * __restrict__ f, Float * __restrict__ fn, dim3 const &fDim,
			     Vec3<Float> const &c, Float c_center,
			     int &j_global,
			     Float &f_center, Float &f_neg_z, Float &f_pos_z,
			     int &fIdx_z)
{
  extern __shared__ float f_shared_[];
  Float *f_shared = reinterpret_cast<Float *>(f_shared_);

  /* NOTE:  Some variable definitions are commented out because they
   * are actually defined in the caller so that they survive over
   * iterations, and passed by reference.  */
  int const y_stride_global = fDim.x;
  int const z_stride_global = fDim.x * fDim.y;
  int const y_stride_shared = blockDim.x + 2;
  // see note above // int j_global = y_stride_global * (blockDim.y * blockIdx.y + threadIdx.y) + (blockDim.x * blockIdx.x + threadIdx.x);
  int const j_shared = y_stride_shared * (threadIdx.y + 1) + (threadIdx.x + 1);
  int const j_shared_neg_x = j_shared - 1;
  int const j_shared_pos_x = j_shared + 1;
  int const j_shared_neg_y = j_shared - y_stride_shared;
  int const j_shared_pos_y = j_shared + y_stride_shared;
  // see note above //  Float f_center = f[j_global];
  // see note above //  Float f_neg_z = f_center; // apply Neumann condition on z- boundary
  // see note above //  Float f_pos_z; // will be assigned later

  /* NOTE:  We do not write to shared memory very first of the
   * function, because no __syncthreads() is issued at the end of each
   * iterations and previous iteration may still be running.  See
   * notes on the bottom of this function.  */

  // NOTE: assuming gridDim.z is always 1
  if (checkBottom && fIdx_z == fDim.z - 1)
    f_pos_z = f_center; // apply Neumann condition on z+ boundary
  else
    f_pos_z = f[j_global + z_stride_global];

  // seems it's safe to write to shared memory now...

#if GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y1 || GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y1Z
  // NOTE: assuming blockDim.y is always 1
  if (blockIdx.y == 0)
    f_shared[j_shared_neg_y] = f_center; // apply Neumann condition on y- boundary
  else
    f_shared[j_shared_neg_y] = f[j_global - y_stride_global];
  if (blockIdx.y == gridDim.y - 1)
    f_shared[j_shared_pos_y] = f_center; // apply Neumann condition on y+ boundary
  else
    f_shared[j_shared_pos_y] = f[j_global + y_stride_global];
#elif GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y2 || GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y2Z
  // NOTE: assuming blockDim.y is always 2
  if (threadIdx.y == 0) {
    if (blockIdx.y == 0)
      f_shared[j_shared_neg_y] = f_center; // apply Neumann condition on y- boundary
    else
      f_shared[j_shared_neg_y] = f[j_global - y_stride_global];
  } else {
    if (blockIdx.y == gridDim.y - 1)
      f_shared[j_shared_pos_y] = f_center; // apply Neumann condition on y+ boundary
    else
      f_shared[j_shared_pos_y] = f[j_global + y_stride_global];
  }
#elif GPU_IMPL_CLASS == GPU_IMPL_CLASS_YN
  // NOTE: assuming no specific blockDim.y
  if (threadIdx.y == 0) {
    if (blockIdx.y == 0)
      f_shared[j_shared_neg_y] = f_center; // apply Neumann condition on y- boundary
    else
      f_shared[j_shared_neg_y] = f[j_global - y_stride_global];
  }
  if (threadIdx.y == blockDim.y - 1) {
    if (blockIdx.y == gridDim.y - 1)
      f_shared[j_shared_pos_y] = f_center; // apply Neumann condition on y+ boundary
    else
      f_shared[j_shared_pos_y] = f[j_global + y_stride_global];
  }
#else
#error improper GPU_IMPL
#endif

  // NOTE: assuming gridDim.x is always 1
  if (threadIdx.x == 0)
    f_shared[j_shared_neg_x] = f_center; // apply Neumann condition on x- boundary
  if (threadIdx.x == blockDim.x - 1)
    f_shared[j_shared_pos_x] = f_center; // apply Neumann condition on x+ boundary

  f_shared[j_shared] = f_center;

  __syncthreads();

#if USE_NUM_OPS_EXPRESSION == 13

  /*
   * Elapsed Time= 3.640e+00 [sec]
   * Performance= 12266.99 [MFlops]
   * Error[128][128][128]=1.3977e-05
   */
  fn[j_global] = (c.x * f_shared[j_shared_neg_x] + c.x * f_shared[j_shared_pos_x] +
		  c.y * f_shared[j_shared_neg_y] + c.y * f_shared[j_shared_pos_y] +
		  c.z * f_neg_z + c.z * f_pos_z + c_center * f_center);

#elif USE_NUM_OPS_EXPRESSION == 10

  // A cheat reducing floating point ops from 13 to 10.
  // This is slightly faster than 13 ops version and increases accuracy.
  /*
   * Elapsed Time= 3.533e+00 [sec]
   * Performance= 9722.91 [MFlops]
   * Error[128][128][128]=1.2980e-05
   */
  fn[j_global] = (c.x * (f_shared[j_shared_neg_x] + f_shared[j_shared_pos_x]) +
		  c.y * (f_shared[j_shared_neg_y] + f_shared[j_shared_pos_y]) +
		  c.z * (f_neg_z + f_pos_z) + c_center * f_center);

#else
#error USE_NUM_OPS_EXPRESSION must be one of [13, 10]
#endif

  j_global += z_stride_global;
  f_neg_z = f_center;
  f_center = f_pos_z;

  fIdx_z++;

  /* BUG:  Following call is intentionally commented out for speed
   * (gain +3%), but behavior depends on hardware and generated code.
   * ********************* IT IS VERY FRAGILE *********************.
   * SDK 4.0 + 8600 GT seems to work with the compiler options in
   * supplied Makefile.  */
  //  __syncthreads();
}

//

#if GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y1 || \
    GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y2 || \
    GPU_IMPL_CLASS == GPU_IMPL_CLASS_YN
// NOTE: assuming gridDim.x and gridDim.z are always 1
template <typename Float>
__device__ void gpu_impl(Float const * __restrict__ f, Float * __restrict__ fn, dim3 const &fDim,
			 Vec3<Float> const &c, Float c_center)
{
  int const y_stride_global = fDim.x;
  int j_global = y_stride_global * (blockDim.y * blockIdx.y + threadIdx.y) + (blockDim.x * blockIdx.x + threadIdx.x);

  Float f_center = f[j_global];
  Float f_neg_z = f_center; // apply Neumann condition on z- boundary
  Float f_pos_z; // will be assigned later

  for (int fIdx_z = 0; fIdx_z < fDim.z; ) {
    gpu_impl_sub<Float, true>(f, fn, fDim, c, c_center, j_global, f_center, f_neg_z, f_pos_z, fIdx_z);
  }
}
#endif

//

#if GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y1Z || GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y2Z
template <typename Float, int num>
__device__ void gpu_impl_sub_repeat(Float const * __restrict__ f, Float * __restrict__ fn, dim3 const &fDim,
				    Vec3<Float> const &c, Float c_center,
				    int &j_global,
				    Float &f_center, Float &f_neg_z, Float &f_pos_z,
				    int &fIdx_z)
{
  // assert(num > 1);
  gpu_impl_sub<Float, false>(f, fn, fDim, c, c_center, j_global, f_center, f_neg_z, f_pos_z, fIdx_z);
  gpu_impl_sub_repeat<Float, num - 1>(f, fn, fDim, c, c_center, j_global, f_center, f_neg_z, f_pos_z, fIdx_z);
}

// C++ doesn't allow partial specialization of functions, so we need instanciate the function manually
#define SUB_REPEAT(FLOAT) \
template <> \
__device__ void gpu_impl_sub_repeat<FLOAT, 1>(FLOAT const * __restrict__ f, FLOAT * __restrict__ fn, dim3 const &fDim, \
					      Vec3<FLOAT> const &c, FLOAT c_center, \
					      int &j_global, \
					      FLOAT &f_center, FLOAT &f_neg_z, FLOAT &f_pos_z, \
					      int &fIdx_z) \
{ \
  gpu_impl_sub<FLOAT, true>(f, fn, fDim, c, c_center, \
		     j_global, f_center, f_neg_z, f_pos_z, fIdx_z); \
}

#ifdef FLOAT
// if FLOAT is defined, assume that is the right type to be used in
// gpu_impl<>() instanciation.
SUB_REPEAT(FLOAT)
#else
// this is more safe (but slow to compile) fallback
SUB_REPEAT(float)
SUB_REPEAT(double)
#endif

// NOTE: assuming gridDim.x and gridDim.z are always 1
// NOTE: in GPU_IMPL_Y1Z, assuming blockDim.y is always 1
// NOTE: in GPU_IMPL_Y2Z, assuming blockDim.y is always 2
// NOTE: in GPU_IMPL_Y1Z<num>, assuming that fDim.z is multiple of <num>.
// NOTE: in GPU_IMPL_Y2Z<num>, assuming that fDim.z is multiple of <num>.
// GPU_IMPL_Y1Z1 is (at least logically) identical to GPU_IMPL_Y1
// GPU_IMPL_Y2Z1 is (at least logically) identical to GPU_IMPL_Y2
template <typename Float>
__device__ void gpu_impl(Float const * __restrict__ f, Float * __restrict__ fn, dim3 const &fDim,
			 Vec3<Float> const &c, Float c_center)
{
  int const y_stride_global = fDim.x;
  int j_global = y_stride_global * (blockDim.y * blockIdx.y + threadIdx.y) + (blockDim.x * blockIdx.x + threadIdx.x);

  Float f_center = f[j_global];
  Float f_neg_z = f_center; // apply Neumann condition on z- boundary
  Float f_pos_z; // will be assigned later

  for (int fIdx_z = 0; fIdx_z < fDim.z; ) {
#if GPU_IMPL_PARAM >= 1
    gpu_impl_sub_repeat<Float, GPU_IMPL_PARAM>(f, fn, fDim, c, c_center, j_global, f_center, f_neg_z, f_pos_z, fIdx_z);
#else
#error improper GPU_IMPL
#endif
  }
}
#endif

//

template <typename Float>
__global__ void gpu_diffusion3d(Float *f, Float *fn, dim3 numNodesInGrid, Vec3<Float> c, Float cc)
{
  // NOTE: if 'error: identifier "gpu_impl" is undefined' reported on
  // following line, check if GPU_IMPL is properly handled by #if's.
  gpu_impl<Float>(f, fn, numNodesInGrid, c, cc);
}

#endif
