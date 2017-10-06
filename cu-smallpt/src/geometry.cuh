#pragma once

//-----------------------------------------------------------------------------
// Includes
//-----------------------------------------------------------------------------
#pragma region

#include "vector.hpp"

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	struct Ray final {

	public:

		//---------------------------------------------------------------------
		// Constructors and Destructors
		//---------------------------------------------------------------------

		__device__ explicit Ray(const Vector3 &o, const Vector3 &d,
			double tmin = 0.0, double tmax = INFINITY, uint32_t depth = 0) noexcept
			: m_o(o), m_d(d),
			m_tmin(tmin), m_tmax(tmax), m_depth(depth) {};
		__device__ explicit Ray(Vector3 &&o, Vector3 &&d,
			double tmin = 0.0, double tmax = INFINITY, uint32_t depth = 0) noexcept
			: m_o(std::move(o)), m_d(std::move(d)),
			m_tmin(tmin), m_tmax(tmax), m_depth(depth) {};
		__device__ Ray(const Ray &ray) noexcept = default;
		__device__ Ray(Ray &&ray) noexcept = default;
		__device__ ~Ray() = default;

		//---------------------------------------------------------------------
		// Assignment Operators
		//---------------------------------------------------------------------

		__device__ Ray &operator=(const Ray &ray) = default;
		__device__ Ray &operator=(Ray &&ray) = default;

		//---------------------------------------------------------------------
		// Member Methods
		//---------------------------------------------------------------------

		__device__ const Vector3 operator()(double t) const {
			return m_o + m_d * t;
		}

		//---------------------------------------------------------------------
		// Member Variables
		//---------------------------------------------------------------------

		Vector3 m_o, m_d;
		mutable double m_tmin, m_tmax;
		uint32_t m_depth;
	};
}