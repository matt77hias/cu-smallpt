#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "math_tools.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <algorithm>

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	//-------------------------------------------------------------------------
	// Declarations and Definitions: Vector3
	//-------------------------------------------------------------------------

	struct Vector3 final {

	public:

		//---------------------------------------------------------------------
		// Constructors and Destructors
		//---------------------------------------------------------------------

		__host__ __device__ explicit Vector3(double a = 0.0) noexcept
			: Vector3(a, a, a) {}
		__host__ __device__ Vector3(double x, double y, double z) noexcept
			: m_x(x), m_y(y), m_z(z) {}
		__host__ __device__ Vector3(const Vector3 &v) noexcept = default;
		__host__ __device__ Vector3(Vector3 &&v) noexcept = default;
		__host__ __device__ ~Vector3() = default;

		//---------------------------------------------------------------------
		// Assignment Operators
		//---------------------------------------------------------------------

		__host__ __device__ Vector3 &operator=(const Vector3 &v) = default;
		__host__ __device__ Vector3 &operator=(Vector3 &&v) = default;

		//---------------------------------------------------------------------
		// Member Methods
		//---------------------------------------------------------------------

		__device__ bool HasNaNs() const noexcept {
			return isnan(m_x) || isnan(m_y) || isnan(m_z);
		}

		__device__ const Vector3 operator-() const noexcept {
			return Vector3(-m_x, -m_y, -m_z);
		}
		
		__device__ const Vector3 operator+(const Vector3 &v) const noexcept {
			return Vector3(m_x + v.m_x, m_y + v.m_y, m_z + v.m_z);
		}
		__device__ const Vector3 operator-(const Vector3 &v) const noexcept {
			return Vector3(m_x - v.m_x, m_y - v.m_y, m_z - v.m_z);
		}
		__device__ const Vector3 operator*(const Vector3 &v) const noexcept {
			return Vector3(m_x * v.m_x, m_y * v.m_y, m_z * v.m_z);
		}
		__device__ const Vector3 operator/(const Vector3 &v) const noexcept {
			return Vector3(m_x / v.m_x, m_y / v.m_y, m_z / v.m_z);
		}
		
		__device__ const Vector3 operator+(double a) const noexcept {
			return Vector3(m_x + a, m_y + a, m_z + a);
		}
		__device__ const Vector3 operator-(double a) const noexcept {
			return Vector3(m_x - a, m_y - a, m_z - a);
		}
		__device__ const Vector3 operator*(double a) const noexcept {
			return Vector3(m_x * a, m_y * a, m_z * a);
		}
		__device__ const Vector3 operator/(double a) const noexcept {
			const double inv_a = 1.0 / a;
			return Vector3(m_x * inv_a, m_y * inv_a, m_z * inv_a);
		}
		
		__device__ friend const Vector3 operator+(
			double a, const Vector3 &v) noexcept {
			
			return Vector3(a + v.m_x, a + v.m_y, a + v.m_z);
		}
		__device__ friend const Vector3 operator-(
			double a, const Vector3 &v) noexcept {
			
			return Vector3(a - v.m_x, a - v.m_y, a - v.m_z);
		}
		__device__ friend const Vector3 operator*(
			double a, const Vector3 &v) noexcept {
			
			return Vector3(a * v.m_x, a * v.m_y, a * v.m_z);
		}
		__device__ friend const Vector3 operator/(
			double a, const Vector3 &v) noexcept {
			
			return Vector3(a / v.m_x, a / v.m_y, a / v.m_z);
		}

		__device__ Vector3 &operator+=(const Vector3 &v) noexcept {
			m_x += v.m_x;
			m_y += v.m_y;
			m_z += v.m_z;
			return *this;
		}
		__device__ Vector3 &operator-=(const Vector3 &v) noexcept {
			m_x -= v.m_x;
			m_y -= v.m_y;
			m_z -= v.m_z;
			return *this;
		}
		__device__ Vector3 &operator*=(const Vector3 &v) noexcept {
			m_x *= v.m_x;
			m_y *= v.m_y;
			m_z *= v.m_z;
			return *this;
		}
		__device__ Vector3 &operator/=(const Vector3 &v) noexcept {
			m_x /= v.m_x;
			m_y /= v.m_y;
			m_z /= v.m_z;
			return *this;
		}
		
		__device__ Vector3 &operator+=(double a) noexcept {
			m_x += a;
			m_y += a;
			m_z += a;
			return *this;
		}
		__device__ Vector3 &operator-=(double a) noexcept {
			m_x -= a;
			m_y -= a;
			m_z -= a;
			return *this;
		}
		__device__ Vector3 &operator*=(double a) noexcept {
			m_x *= a;
			m_y *= a;
			m_z *= a;
			return *this;
		}
		__device__ Vector3 &operator/=(double a) noexcept {
			const double inv_a = 1.0 / a;
			m_x *= inv_a;
			m_y *= inv_a;
			m_z *= inv_a;
			return *this;
		}

		__device__ double Dot(const Vector3 &v) const noexcept {
			return m_x * v.m_x + m_y * v.m_y + m_z * v.m_z;
		}
		__device__ const Vector3 Cross(const Vector3 &v) const noexcept {
			return Vector3(m_y * v.m_z - m_z * v.m_y, m_z * v.m_x - m_x * v.m_z, m_x * v.m_y - m_y * v.m_x);
		}

		__device__ bool operator==(const Vector3 &v) const {
			return m_x == v.m_x && m_y == v.m_y && m_z == v.m_z;
		}
		__device__ bool operator!=(const Vector3 &v) const {
			return m_x != v.m_x || m_y != v.m_y || m_z != v.m_z;
		}
		__device__ bool operator<(const Vector3 &v) const {
			return m_x < v.m_x && m_y < v.m_y && m_z < v.m_z;
		}
		__device__ bool operator<=(const Vector3 &v) const {
			return m_x <= v.m_x && m_y <= v.m_y && m_z <= v.m_z;
		}
		__device__ bool operator>(const Vector3 &v) const {
			return m_x > v.m_x && m_y > v.m_y && m_z > v.m_z;
		}
		__device__ bool operator>=(const Vector3 &v) const {
			return m_x >= v.m_x && m_y >= v.m_y && m_z >= v.m_z;
		}

		__device__ double operator[](size_t i) const noexcept {
			return (&m_x)[i];
		}
		__device__ double &operator[](size_t i) noexcept {
			return (&m_x)[i];
		}

		__device__ size_t MinDimension() const noexcept {
			return (m_x < m_y && m_x < m_z) ? 0 : ((m_y < m_z) ? 1 : 2);
		}
		__device__ size_t MaxDimension() const noexcept {
			return (m_x > m_y && m_x > m_z) ? 0 : ((m_y > m_z) ? 1 : 2);
		}
		__device__ double Min() const noexcept {
			return (m_x < m_y && m_x < m_z) ? m_x : ((m_y < m_z) ? m_y : m_z);
		}
		__device__ double Max() const noexcept {
			return (m_x > m_y && m_x > m_z) ? m_x : ((m_y > m_z) ? m_y : m_z);
		}

		__device__ double Norm2_squared() const noexcept {
			return m_x * m_x + m_y * m_y + m_z * m_z;
		}
		__device__ double Norm2() const noexcept {
			return sqrt(Norm2_squared());
		}
		__device__ void Normalize() noexcept {
			const double a = 1.0 / Norm2();
			m_x *= a;
			m_y *= a;
			m_z *= a;
		}

		//---------------------------------------------------------------------
		// Member Variables
		//---------------------------------------------------------------------

		double m_x, m_y, m_z;
	};
	
	//-------------------------------------------------------------------------
	// Declarations and Definitions: Vector3 Utilities
	//-------------------------------------------------------------------------

	__device__ inline const Vector3 Sqrt(const Vector3 &v) noexcept {
		return Vector3(sqrt(v.m_x), 
			           sqrt(v.m_y), 
			           sqrt(v.m_z));
	}
	
	__device__ inline const Vector3 Pow(const Vector3 &v, double a) noexcept {
		return Vector3(pow(v.m_x, a), 
			           pow(v.m_y, a), 
			           pow(v.m_z, a));
	}
	
	__device__ inline const Vector3 Abs(const Vector3 &v) noexcept {
		return Vector3(abs(v.m_x), 
			           abs(v.m_y), 
			           abs(v.m_z));
	}
	
	__device__ inline const Vector3 Min(const Vector3 &v1, const Vector3 &v2) noexcept {
		return Vector3(fmin(v1.m_x, v2.m_x), 
			           fmin(v1.m_y, v2.m_y), 
			           fmin(v1.m_z, v2.m_z));
	}
	
	__device__ inline const Vector3 Max(const Vector3 &v1, const Vector3 &v2) noexcept {
		return Vector3(fmax(v1.m_x, v2.m_x), 
			           fmax(v1.m_y, v2.m_y), 
			           fmax(v1.m_z, v2.m_z));
	}
	
	__device__ inline const Vector3 Round(const Vector3 &v) noexcept {
		return Vector3(round(v.m_x), 
			           round(v.m_y), 
			           round(v.m_z));
	}
	
	__device__ inline const Vector3 Floor(const Vector3 &v) noexcept {
		return Vector3(floor(v.m_x), 
			           floor(v.m_y), 
			           floor(v.m_z));
	}
	
	__device__ inline const Vector3 Ceil(const Vector3 &v) noexcept {
		return Vector3(ceil(v.m_x),
			           ceil(v.m_y), 
			           ceil(v.m_z));
	}
	
	__device__ inline const Vector3 Trunc(const Vector3 &v) noexcept {
		return Vector3(trunc(v.m_x), 
			           trunc(v.m_y), 
			           trunc(v.m_z));
	}
	
	__device__ inline const Vector3 Clamp(
		const Vector3 &v, double low = 0.0, double high = 1.0) noexcept {
		
		return Vector3(Clamp(v.m_x, low, high), 
			           Clamp(v.m_y, low, high), 
			           Clamp(v.m_z, low, high));
	}
	
	__device__ inline const Vector3 Lerp(
		double a, const Vector3 &v1, const Vector3 &v2) noexcept {
		
		return v1 + a * (v2 - v1);
	}
	
	__device__ inline const Vector3 Permute(
		const Vector3 &v, size_t x, size_t y, size_t z) noexcept {
		
		return Vector3(v[x], v[y], v[z]);
	}

	__device__ inline const Vector3 Normalize(const Vector3 &v) noexcept {
		const double a = 1.0 / v.Norm2();
		return a * v;
	}
}