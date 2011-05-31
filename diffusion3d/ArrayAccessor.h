#ifndef ArrayAccessor3_h__
#define ArrayAccessor3_h__ 1

#include "Vec.h"

template <typename T>
class ArrayAccessor3
{
public:
  ArrayAccessor3(Dim3 const &dim, T *array) :
    dim_(dim),
    array_(array)
  {
  }

  int index(int x, int y, int z) const
  {
    return dim_.x * dim_.y * z + dim_.x * y + x;
  }

  int index(int3 const &x) const
  {
    return index(x.x, x.y, x.z);
  }

  T const & operator[](int i) const
  {
    return array_[i];
  }

  T & operator[](int i)
  {
    return array_[i];
  }

  T const & operator()(int i) const
  {
    return array_[i];
  }

  T & operator()(int i)
  {
    return array_[i];
  }

  T & operator()(int x, int y, int z)
  {
    return array_[index(x, y, z)];
  }

  T & operator()(int3 const &x)
  {
    return array_[index(x)];
  }

private:
  Dim3 const dim_;
  T * const array_;
};

#endif
