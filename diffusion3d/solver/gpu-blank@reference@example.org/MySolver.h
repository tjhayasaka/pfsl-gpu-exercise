//

#ifndef MySolver_h__
#define MySolver_h__ 1

#include "Solver.h"
#include "ArrayAccessor.h"
#include <boost/scoped_array.hpp>

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
  boost::scoped_array<Float> p(new Float[this->numTotalNodesInGridIncludingHalo()]);

  if (mode != Solver<Float>::bom_wo) {
    cudaMemcpy(p.get(), buf_front_.get(), sizeof(Float) * this->numTotalNodesInGridIncludingHalo(), cudaMemcpyDeviceToHost);
    assert(cudaGetLastError() == cudaSuccess);
  }

  func(*this, p.get());

  if (mode != Solver<Float>::bom_ro) {
    cudaMemcpy(buf_front_.get(), p.get(), sizeof(Float) * this->numTotalNodesInGridIncludingHalo(), cudaMemcpyHostToDevice);
    assert(cudaGetLastError() == cudaSuccess);
  }
}

template <typename Float>
__global__ void gpu_diffusion3d(Float *f, Float *fn, dim3 numNodesInGrid, Vec3<Float> c, Float cc)
{
  /*
   * your code comes here
   */
}

template <typename Float>
int MySolver<Float>::nextTick(Float kappa, Float dt)
{
  uint const nx = this->numNodesInGrid().x;
  uint const ny = this->numNodesInGrid().y;
  //uint const nz = this->numNodesInGrid().z;
  Vec3<Float> const c = kappa * dt / (cellSpacing_ * cellSpacing_);
  Float const cc = Float(1.0) - (c.x + c.x + c.y + c.y + c.z + c.z);

  /*
   * your code comes here
   */
  //  gpu_diffusion3d<Float><<< ?, ?, ? >>>(buf_front_.get(), buf_back_.get(), numNodesInGrid_, c, cc);
  /*
   * your code comes here
   */

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
