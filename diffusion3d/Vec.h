//

#ifndef Vec_h__
#define Vec_h__ 1

#include <vector_types.h>

template <typename T>
struct Vec3__ {
private: // only limited set of T is allowed.  see bellow for available types
  Vec3__();
  Vec3__(T x, T y, T z);
};

// available types and constructors

template <> struct Vec3__<unsigned int> : public dim3
{ Vec3__(unsigned int x_, unsigned int y_, unsigned int z_) { x = x_; y = y_; z = z_; } };

template <> struct Vec3__<int> : public int3
{ Vec3__(int x_, int y_, int z_) { x = x_; y = y_; z = z_; } };

template <> struct Vec3__<float> : public float3
{ Vec3__(float x_, float y_, float z_) { x = x_; y = y_; z = z_; } };

template <> struct Vec3__<double> : public double3
{ Vec3__(double x_, double y_, double z_) { x = x_; y = y_; z = z_; } };

//

template <typename T>
struct Vec3 : public Vec3__<T>
{
  Vec3() : Vec3__<T>()
  {}

  template <typename U>
  Vec3(U x, U y, U z) : Vec3__<T>(x, y, z)
  {}

  template <typename U>
  Vec3(Vec3<U> const &x) : Vec3__<T>(x.x, x.y, x.z)
  {}

  Vec3 & operator *=(T const &v1) { this->x *= v1; this->y *= v1; this->z *= v1; return *this; }
  Vec3 & operator /=(T const &v1) { this->x /= v1; this->y /= v1; this->z /= v1; return *this; }
  Vec3 & operator *=(Vec3 const &v1) { this->x *= v1.x; this->y *= v1.y; this->z *= v1.z; return *this; }
  Vec3 & operator /=(Vec3 const &v1) { this->x /= v1.x; this->y /= v1.y; this->z /= v1.z; return *this; }
  Vec3 & operator +=(Vec3 const &v1) { this->x += v1.x; this->y += v1.y; this->z += v1.z; return *this; }
  Vec3 & operator -=(Vec3 const &v1) { this->x -= v1.x; this->y -= v1.y; this->z -= v1.z; return *this; }
};

template <typename T>
static inline Vec3<T> operator *(Vec3<T> v0, T const &v1) { return v0 *= v1; }
template <typename T>
static inline Vec3<T> operator *(Vec3<T> v0, Vec3<T> const &v1) { return v0 *= v1; }
template <typename T>
static inline Vec3<T> operator /(Vec3<T> v0, T const &v1) { return v0 /= v1; }
template <typename T>
static inline Vec3<T> operator /(Vec3<T> v0, Vec3<T> const &v1) { return v0 /= v1; }
template <typename T>
static inline Vec3<T> operator +(Vec3<T> v0, Vec3<T> const &v1) { return v0 += v1; }
template <typename T>
static inline Vec3<T> operator -(Vec3<T> v0, Vec3<T> const &v1) { return v0 -= v1; }

template <typename T>
static inline Vec3<T> operator *(T const &v0, Vec3<T> v1)
{
  return v1 *= v0;
}

template <typename T>
static inline Vec3<T> operator /(T const &v0, Vec3<T> v1)
{
  v1.x = v0 / v1.x;
  v1.y = v0 / v1.y;
  v1.z = v0 / v1.z;
  return v1;
}

typedef Vec3<unsigned int> Dim3;
typedef Vec3<int> Int3;

#endif
