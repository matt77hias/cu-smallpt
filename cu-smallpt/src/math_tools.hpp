#pragma once

//-----------------------------------------------------------------------------
// CUDA Includes
//-----------------------------------------------------------------------------
#pragma region

#include "device_launch_parameters.h"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <math.h>
#include <stdint.h>

#pragma endregion

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#pragma region

#define M_PI 3.14159265358979323846

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	__host__ __device__ inline double Clamp(
		double x, double low = 0, double high = 1) noexcept {
		return (x < high) ? ((x > low) ? x : low) : high;
	}

	inline uint8_t ToByte(double x, double gamma = 2.2) noexcept {
		return static_cast< uint8_t >(Clamp(255.0 * pow(x, 1 / gamma), 
			                                0.0, 255.0));
	}
}