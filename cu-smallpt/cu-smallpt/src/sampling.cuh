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

	__device__ inline Vector3 UniformSampleOnHemisphere(double u1, 
														double u2) {
		// u1 := cos_theta
		const double sin_theta = std::sqrt(fmax(0.0, 1.0 - u1 * u1));
		const double phi = 2.0 * g_pi * u2;
		return {
			std::cos(phi) * sin_theta,
			std::sin(phi) * sin_theta,
			u1
		};
	}

	__device__ inline Vector3 CosineWeightedSampleOnHemisphere(double u1, 
															   double u2) {
		const double cos_theta = sqrt(1.0 - u1);
		const double sin_theta = sqrt(u1);
		const double phi = 2.0 * g_pi * u2;
		return {
			std::cos(phi) * sin_theta,
			std::sin(phi) * sin_theta,
			cos_theta
		};
	}
}