//

#ifndef MySolver_h__
#define MySolver_h__ 1

#include "Solver.h"
#include "ArrayAccessor.h"
#include <boost/scoped_array.hpp>

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
  virtual int halo() const { return 1; }
  virtual Dim3 const &numNodesInGrid() const { return numNodesInGrid_; }
  virtual Vec3<Float> const &cellSpacing() const { return cellSpacing_; }

private:
  Dim3 numNodesInGrid_;
  Vec3<Float> cellSpacing_;
  boost::scoped_array<Float> buf_back_;
  boost::scoped_array<Float> buf_front_;
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
  buf_back_.reset(new Float[this->numTotalNodesInGridIncludingHalo()]);
  buf_front_.reset(new Float[this->numTotalNodesInGridIncludingHalo()]);
}

template <typename Float>
void MySolver<Float>::withFrontBuffer(enum Solver<Float>::BufferOpMode mode, boost::function<void (Solver<Float> const &solver, Float *f)> func)
{
  func(*this, buf_front_.get());
}

template <typename Float>
int MySolver<Float>::nextTick(Float kappa, Float dt)
{
#ifndef USE_NUM_OPS_EXPRESSION
#define USE_NUM_OPS_EXPRESSION 13
#endif

  ArrayAccessor3<Float> f(this->numNodesInGridIncludingHalo(), buf_front_.get());
  ArrayAccessor3<Float> fn(this->numNodesInGridIncludingHalo(), buf_back_.get());
  int const nx = this->numNodesInGrid().x;
  int const ny = this->numNodesInGrid().y;
  int const nz = this->numNodesInGrid().z;
  Vec3<Float> const c = kappa * dt / (cellSpacing_ * cellSpacing_);
  Float const cc = Float(1.0) - (c.x + c.x + c.y + c.y + c.z + c.z);
  int const ix = f.index(1, 0, 0);
  int const iy = f.index(0, 1, 0);
  int const iz = f.index(0, 0, 1);

  // diffuse

  for (int jz = 1; jz < nz + 1; jz++) {
    for (int jy = 1; jy < ny + 1; jy++) {
      for (int jx = 1; jx < nx + 1; jx++) {
	int const j = f.index(jx, jy, jz);
#if USE_NUM_OPS_EXPRESSION == 13

#if 0
	// A standard 13 ops expression.  Not fast.
	/*
	 * Elapsed Time= 1.390e+00 [sec]
	 * Performance= 1005.09 [MFlops]
	 * Error[64][64][64]=1.7751e-05
	 *
	 * Elapsed Time= 4.562e+01 [sec]
	 * Performance=  978.91 [MFlops]
	 * Error[128][128][128]=4.3725e-06
	 */
	fn[j] = (cc * f[j] +
		 c.x * f[j - ix] + c.x * f[j + ix] +
		 c.y * f[j - iy] + c.y * f[j + iy] +
		 c.z * f[j - iz] + c.z * f[j + iz]);
#else
	// I think this is not prettier than previous one, but 35%
	// faster (tested with gcc-4.4 -O3, C2Q 9550).
	// Despite of the high ops count (7 fmul + 6 fadd, you can
	// look into disassembly list to confirm it), this is the
	// fastest cpu implementation of this example,
	/*
	 * Elapsed Time= 8.321e-01 [sec]
	 * Performance= 1679.24 [MFlops]
	 * Error[64][64][64]=1.7751e-05
	 *
	 * Elapsed Time= 2.763e+01 [sec]
	 * Performance= 1616.21 [MFlops]
	 * Error[128][128][128]=4.3725e-06
	 */
	fn[j] = (c.z * f[j + iz] + c.z * f[j - iz] +
		 c.y * f[j + iy] + c.y * f[j - iy] +
		 c.x * f[j + ix] + cc * f[j] + c.x * f[j - ix]);
#endif

#elif USE_NUM_OPS_EXPRESSION == 11

	// A cheat reducing floating point ops from 13 to 11.
	// Unfortunately not faster than optimized 13.
	/*
	 * Elapsed Time= 8.393e-01 [sec]
	 * Performance= 1408.60 [MFlops]
	 * Error[64][64][64]=1.7751e-05
	 *
	 * Elapsed Time= 2.785e+01 [sec]
	 * Performance= 1356.77 [MFlops]
	 * Error[128][128][128]=4.3725e-06
	 */
	fn(j) = (c.z * (f(j + iz) + f(j - iz)) +
		 c.y * (f(j + iy) + f(j - iy)) +
		 c.x * f(j + ix) + cc * f(j) + c.x * f(j - ix));

#elif USE_NUM_OPS_EXPRESSION == 10

	// A cheat reducing floating point ops from 13 to 10.
	// Unfortunately not faster than optimized 13 or 11.  The
	// weired order of sub-expressions affect the speed.
	/*
	 * Elapsed Time= 9.340e-01 [sec]
	 * Performance= 1150.71 [MFlops]
	 * Error[64][64][64]=1.7751e-05
	 *
	 * Elapsed Time= 3.112e+01 [sec]
	 * Performance= 1103.97 [MFlops]
	 * Error[128][128][128]=4.3725e-06
	 */
	fn(j) = (c.z * (f(j - iz) + f(j + iz)) +
		 cc * f(j) +
		 c.y * (f(j - iy) + f(j + iy)) +
		 c.x * (f(j - ix) + f(j + ix)));

#else
#error USE_NUM_OPS_EXPRESSION must be one of [13, 11, 10]
#endif
      }
    }
  }

  // apply wall boundary condition

  for (int jz = 1; jz < nz + 1; jz++) {
    for (int jy = 1; jy < ny + 1; jy++) {
      int const j0 = fn.index(0, jy, jz);
      fn(j0) = fn(j0 + ix);
      int const j1 = fn.index(nx + 1, jy, jz);
      fn(j1) = fn(j1 - ix);
    }
  }

  for (int jz = 1; jz < nz + 1; jz++) {
    for (int jx = 1; jx < nx + 1; jx++) {
      int const j0 = fn.index(jx, 0, jz);
      fn(j0) = fn(j0 + iy);
      int const j1 = fn.index(jx, ny + 1, jz);
      fn(j1) = fn(j1 - iy);
    }
  }

  for (int jy = 1; jy < ny + 1; jy++) {
    for (int jx = 1; jx < nx + 1; jx++) {
      int const j0 = fn.index(jx, jy, 0);
      fn(j0) = fn(j0 + iz);
      int const j1 = fn.index(jx, jy, nz + 1);
      fn(j1) = fn(j1 - iz);
    }
  }

  return this->numTotalNodesInGrid() * USE_NUM_OPS_EXPRESSION;
}

template <typename Float>
void MySolver<Float>::swapBuffers()
{
  boost::swap(buf_back_, buf_front_);
}

#endif
