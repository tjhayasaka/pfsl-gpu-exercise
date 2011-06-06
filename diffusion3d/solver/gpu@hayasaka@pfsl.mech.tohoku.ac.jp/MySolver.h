//

#ifndef MySolver_h__
#define MySolver_h__ 1

#include "Solver.h"
#include "ArrayAccessor.h"
#include <boost/scoped_array.hpp>

//

#ifndef USE_NUM_OPS_EXPRESSION
#define USE_NUM_OPS_EXPRESSION 13
#endif

#include "gpu_impl.h"

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
  virtual int halo() const { return 0; }
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

template <typename Float>
int MySolver<Float>::nextTick(Float kappa, Float dt)
{
  uint const nx = this->numNodesInGrid().x;
  uint const ny = this->numNodesInGrid().y;
  //uint const nz = this->numNodesInGrid().z;
  Vec3<Float> const c = kappa * dt / (cellSpacing_ * cellSpacing_);
  Float const cc = Float(1.0) - (c.x + c.x + c.y + c.y + c.z + c.z);

#if GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y1 || GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y1Z
  dim3 threadsInBlock(nx, 1, 1);
#elif GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y2 || GPU_IMPL_CLASS == GPU_IMPL_CLASS_Y2Z
  dim3 threadsInBlock(nx, 2, 1);
#elif GPU_IMPL_CLASS == GPU_IMPL_CLASS_YN
  dim3 threadsInBlock(nx, GPU_IMPL_PARAM, 1);
#else
#error unknown GPU_IMPL is selected
#endif
  dim3 blocksInGrid(1, ny / threadsInBlock.y, 1);
  assert(blocksInGrid.y * threadsInBlock.y == ny);
  gpu_diffusion3d<Float><<< blocksInGrid, threadsInBlock, sizeof(Float) * (nx + 2) * (threadsInBlock.y + 2) >>>(buf_front_.get(), buf_back_.get(), numNodesInGrid_, c, cc);

  cudaThreadSynchronize();
  assert(cudaGetLastError() == cudaSuccess);

  return this->numTotalNodesInGrid() * USE_NUM_OPS_EXPRESSION;
}

template <typename Float>
void MySolver<Float>::swapBuffers()
{
  buf_back_.swap(buf_front_);
}

#endif
