//

#ifndef Solver_h__
#define Solver_h__ 1

#include "Vec.h"
#include <boost/function.hpp>

template <typename Float>
class Solver;

template <typename Float>
class Solver
{
public:
  enum BufferOpMode { bom_ro, bom_wo, bom_rw };

  Solver() {}
  virtual ~Solver() {}
  virtual void reset(Dim3 const &numNodesInGrid, Vec3<Float> const &cellSpacing) = 0;
  virtual void withFrontBuffer(BufferOpMode mode, boost::function<void (Solver<Float> const &solver, Float *buffer)> func) = 0;
  virtual int nextTick(Float kappa, Float dt) = 0;
  virtual void swapBuffers() = 0;
  virtual Float const *buf_back() const = 0;
  virtual Float const *buf_front() const = 0;
  virtual int halo() const = 0;
  virtual Dim3 const &numNodesInGrid() const = 0;
  virtual Dim3 numNodesInGridIncludingHalo() const
  {
    return numNodesInGrid() + Dim3(2, 2, 2) * (unsigned int)halo();
  }
  virtual int numTotalNodesInGrid() const
  {
    Dim3 n = numNodesInGrid();
    return n.x * n.y * n.z;
  }
  virtual int numTotalNodesInGridIncludingHalo() const
  {
    Dim3 n = numNodesInGridIncludingHalo();
    return n.x * n.y * n.z;
  }
  virtual Vec3<Float> const &cellSpacing() const = 0;
private:
};

#endif
