//

#ifndef MySolver_h__
#define MySolver_h__ 1

#include "Solver.h"
#include "ArrayAccessor.h"
#include <boost/scoped_array.hpp>

#define SolverMode 2

//

template <typename T>
class ScopedCudaArray
{
public:
  ScopedCudaArray() :
    data_(NULL)
  {
  }

  ~ScopedCudaArray()
  {
    cudaFree(data_);
    data_ = NULL;
  }

  void reset(int numItems)
  {
    cudaFree(data_);
    data_ = NULL;
    cudaMalloc(&data_, sizeof(T) * numItems);
    assert(cudaGetLastError() == cudaSuccess);
  }

  T const *get() const { return data_; }
  T *get() { return data_; }
  void swap(ScopedCudaArray<T> &o) { std::swap(data_, o.data_); }

private:
  T *data_;
};


//

template <typename Float>
class MySolver : public Solver<Float>
{
public:
  MySolver();
  virtual ~MySolver();
  virtual void reset(Dim3 const &numNodesInGrid, Vec3<Float> const &cellSpacing);
  virtual void withFrontBuffer(enum Solver<Float>::BufferOpMode mode, boost::function<void (Solver<Float> const &solver, Float *buffer)> func);
  virtual int nextTick(Float kappa, Float dt);
  virtual void swapBuffers();
  virtual Float const *buf_back() const { return buf_back_.get(); }
  virtual Float const *buf_front() const { return buf_front_.get(); }
  virtual int halo() const { return 0; } /* NOTE: may be inappropriate for you  */
  virtual Dim3 const &numNodesInGrid() const { return numNodesInGrid_; }
  virtual Vec3<Float> const &cellSpacing() const { return cellSpacing_; }

private:
  Dim3 numNodesInGrid_;
  Vec3<Float> cellSpacing_;
  ScopedCudaArray<Float> buf_back_;
  ScopedCudaArray<Float> buf_front_;
};

template <typename Float>
MySolver<Float>::MySolver() :
  Solver<Float>(),
  numNodesInGrid_(0, 0, 0),
  cellSpacing_(0, 0, 0)
{
}

template <typename Float>
MySolver<Float>::~MySolver()
{
}

template <typename Float>
void MySolver<Float>::reset(Dim3 const &numNodesInGrid, Vec3<Float> const &cellSpacing)
{
  numNodesInGrid_ = numNodesInGrid;
  cellSpacing_ = cellSpacing;

  buf_back_.reset(this->numTotalNodesInGridIncludingHalo());
  buf_front_.reset(this->numTotalNodesInGridIncludingHalo());
}

template <typename Float>
void MySolver<Float>::withFrontBuffer(enum Solver<Float>::BufferOpMode mode, boost::function<void (Solver<Float> const &solver, Float *f)> func)
{
  boost::scoped_array<Float> host_tmp_p(new Float[this->numTotalNodesInGridIncludingHalo()]);

  if (mode != Solver<Float>::bom_wo) {
    cudaMemcpy(host_tmp_p.get(), buf_front_.get(), sizeof(Float) * this->numTotalNodesInGridIncludingHalo(), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
  }

  func(*this, host_tmp_p.get());

  if (mode != Solver<Float>::bom_ro) {
    cudaMemcpy(buf_front_.get(), host_tmp_p.get(), sizeof(Float) * this->numTotalNodesInGridIncludingHalo(), cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);
  }
}

// SolverMode 0 -------
template <typename Float>
__global__ void gpu_diffusion3d_0(Float *f, Float *fn, dim3 NiG, Vec3<Float> c, Float cc)
{
  int   j, jx, jy, jz;
  Float fcc, fxm, fxp, fym, fyp, fzm, fzp;

  jx = blockDim.x*blockIdx.x + threadIdx.x;
  jy = blockDim.y*blockIdx.y + threadIdx.y;
  jz = blockDim.z*blockIdx.z + threadIdx.z;
  j  = jx + NiG.x*jy + NiG.x*NiG.y*jz;

  fcc = f[j];

  if(jx == 0) fxm = fcc;
  else        fxm = f[j-1];

  if(jx == NiG.x-1) fxp = fcc;
  else              fxp = f[j+1];

  if(jy == 0) fym = fcc;
  else        fym = f[j-NiG.x];

  if(jy == NiG.y-1) fyp = fcc;
  else              fyp = f[j+NiG.x];

  if(jz == 0) fzm = fcc;
  else        fzm = f[j-NiG.x*NiG.y];

  if(jz == NiG.z-1) fzp = fcc;
  else              fzp = f[j+NiG.x*NiG.y];

  fn[j] = (cc*fcc
        +  c.x*fxm + c.x*fxp
        +  c.y*fym + c.y*fyp
        +  c.z*fzm + c.z*fzp);
}

// SolverMode 1 -------
template <typename Float>
__global__ void gpu_diffusion3d_1(Float *f, Float *fn, dim3 NiG, Vec3<Float> c, Float cc)
{
  extern __shared__ float fShd_[];
  Float *fShd = reinterpret_cast<Float *>(fShd_);

  int const yAcsGbl = NiG.x;
  int const zAcsGbl = NiG.x*NiG.y;
  int const yAcsShd = blockDim.x+2;
  int const zAcsShd = (blockDim.x+2)*(blockDim.y+2);

  int const jGbl    = (blockDim.x*blockIdx.x + threadIdx.x)
                    + (blockDim.y*blockIdx.y + threadIdx.y)*yAcsGbl
                    + (blockDim.z*blockIdx.z + threadIdx.z)*zAcsGbl;
  int const jShd    = (threadIdx.x+1)
                    + (threadIdx.y+1)*yAcsShd
                    + (threadIdx.z+1)*zAcsShd;
  int const jShd_xm = jShd - 1;
  int const jShd_xp = jShd + 1;
  int const jShd_ym = jShd - yAcsShd;
  int const jShd_yp = jShd + yAcsShd;
  int const jShd_zm = jShd - zAcsShd;
  int const jShd_zp = jShd + zAcsShd;

  Float fcc = f[jGbl];

  if(threadIdx.x == 0){
    if(blockIdx.x == 0) fShd[jShd_xm] = fcc;
    else                fShd[jShd_xm] = f[jGbl-1];
  }

  if(threadIdx.x == blockDim.x-1){
    if(blockIdx.x == gridDim.x-1) fShd[jShd_xp] = fcc;
    else                          fShd[jShd_xp] = f[jGbl+1];
  }

  if(threadIdx.y == 0){
    if(blockIdx.y == 0) fShd[jShd_ym] = fcc;
    else                fShd[jShd_ym] = f[jGbl-yAcsGbl];
  }

  if(threadIdx.y == blockDim.y-1){
    if(blockIdx.y == gridDim.y-1) fShd[jShd_yp] = fcc;
    else                          fShd[jShd_yp] = f[jGbl+yAcsGbl];
  }

  if(threadIdx.z == 0){
    if(blockIdx.z == 0) fShd[jShd_zm] = fcc;
    else                fShd[jShd_zm] = f[jGbl-zAcsGbl];
  }

  if(threadIdx.z == blockDim.z-1){
    if(blockIdx.z == gridDim.z-1) fShd[jShd_zp] = fcc;
    else                          fShd[jShd_zp] = f[jGbl+zAcsGbl];
  }

  fShd[jShd] = fcc;

  __syncthreads();

  fn[jGbl] = cc*fcc
           + c.x*fShd[jShd_xm] + c.x*fShd[jShd_xp]
           + c.y*fShd[jShd_ym] + c.y*fShd[jShd_yp]
           + c.z*fShd[jShd_zm] + c.z*fShd[jShd_zp];
}

// SolverMode 2 -------
template <typename Float>
__device__ void gpu_diffusion3d_sub(Float const * __restrict__ f, Float * __restrict__ fn,
                                    dim3 const &NiG, Vec3<Float> const &c, Float cc,
                                    int &jGbl, Float &fcc, Float &fzm, Float &fzp, int &Idx_z)
{
  extern __shared__ float fShd_[];
  Float *fShd = reinterpret_cast<Float *>(fShd_);

  int const yAcsGbl = NiG.x;
  int const zAcsGbl = NiG.x*NiG.y;
  int const yAcsShd = blockDim.x+2;

  int const jShd    = (threadIdx.x+1) + yAcsShd*(threadIdx.y+1);
  int const jShd_xm = jShd - 1;
  int const jShd_xp = jShd + 1;
  int const jShd_ym = jShd - yAcsShd;
  int const jShd_yp = jShd + yAcsShd;

  if(Idx_z == NiG.z-1) fzp = fcc;
  else                 fzp = f[jGbl + zAcsGbl];

  if(blockIdx.y == 0) fShd[jShd_ym] = fcc;
  else                fShd[jShd_ym] = f[jGbl - yAcsGbl];

  if(blockIdx.y == gridDim.y-1) fShd[jShd_yp] = fcc;
  else                          fShd[jShd_yp] = f[jGbl + yAcsGbl];

  if(threadIdx.x == 0)            fShd[jShd_xm] = fcc;
  if(threadIdx.x == blockDim.x-1) fShd[jShd_xp] = fcc;

  fShd[jShd] = fcc;

  __syncthreads();

  fn[jGbl] = cc*fcc
           + c.x*fShd[jShd_xm] + c.x*fShd[jShd_xp]
           + c.y*fShd[jShd_ym] + c.y*fShd[jShd_yp]
           + c.z*fzm           + c.z*fzp;

  jGbl += zAcsGbl;
  fzm = fcc;
  fcc = fzp;
}

template <typename Float>
__global__ void gpu_diffusion3d_2(Float *f, Float *fn, dim3 NiG, Vec3<Float> c, Float cc)
{
  int const yAcsGbl = NiG.x;
  int       jGbl = (blockDim.x*blockIdx.x + threadIdx.x)
                 + (blockDim.y*blockIdx.y + threadIdx.y)*yAcsGbl;

  Float fcc = f[jGbl];
  Float fzm = fcc;
  Float fzp;

  for(int Idx_z = 0;Idx_z < NiG.z;Idx_z++)
    gpu_diffusion3d_sub<Float>(f, fn, NiG, c, cc, jGbl, fcc, fzm, fzp, Idx_z);

}
// --------------------

template <typename Float>
int MySolver<Float>::nextTick(Float kappa, Float dt)
{
  uint const nx = this->numNodesInGrid().x;
  uint const ny = this->numNodesInGrid().y;
  uint const nz = this->numNodesInGrid().z;
  Vec3<Float> const c = kappa * dt / (cellSpacing_ * cellSpacing_);
  Float const cc = Float(1.0) - (c.x + c.x + c.y + c.y + c.z + c.z);

#if SolverMode == 0
  dim3 threadsInBlock(32, 16, 1);
  dim3 blocksInGrid(nx/threadsInBlock.x, ny/threadsInBlock.y, nz/threadsInBlock.z);

  gpu_diffusion3d_0<Float>
  <<< blocksInGrid, threadsInBlock >>>
  (buf_front_.get(), buf_back_.get(), numNodesInGrid_, c, cc);
#elif SolverMode == 1
  dim3 threadsInBlock(32, 16, 1);
  dim3 blocksInGrid(nx/threadsInBlock.x, ny/threadsInBlock.y, nz/threadsInBlock.z);

  gpu_diffusion3d_1<Float>
  <<< blocksInGrid, threadsInBlock, sizeof(Float)*(threadsInBlock.x+2)*(threadsInBlock.y+2)*(threadsInBlock.z+2) >>>
  (buf_front_.get(), buf_back_.get(), numNodesInGrid_, c, cc);
#elif SolverMode == 2
  dim3 threadsInBlock(nx, 1, 1);
  dim3 blocksInGrid(1, ny/threadsInBlock.y, 1);

  gpu_diffusion3d_2<Float>
  <<< blocksInGrid, threadsInBlock, sizeof(Float)*(threadsInBlock.x+2)*(threadsInBlock.y+2) >>>
  (buf_front_.get(), buf_back_.get(), numNodesInGrid_, c, cc);
#endif

  cudaThreadSynchronize();
  assert(cudaGetLastError() == cudaSuccess);

  return this->numTotalNodesInGrid() * 13;
}

template <typename Float>
void MySolver<Float>::swapBuffers()
{
  buf_back_.swap(buf_front_);
}

#endif
