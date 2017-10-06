#pragma once

//-----------------------------------------------------------------------------
// CUDA Includes
//-----------------------------------------------------------------------------
#pragma region

#include "cuda_runtime.h"

#pragma endregion

//-----------------------------------------------------------------------------
// System Includes
//-----------------------------------------------------------------------------
#pragma region

#include <stdio.h>
#include <stdlib.h>

#pragma endregion

//-----------------------------------------------------------------------------
// Declarations and Definitions
//-----------------------------------------------------------------------------
namespace smallpt {

	static void HandleError(cudaError_t err, const char *file, int line) {
		if (err != cudaSuccess) {
			printf("%s in %s at line %d\n", 
				cudaGetErrorString(err), file, line);
			exit(EXIT_FAILURE);
		}
	}
}

//-----------------------------------------------------------------------------
// Defines
//-----------------------------------------------------------------------------
#pragma region

#define HANDLE_ERROR(err) (HandleError( err, __FILE__, __LINE__ ))


#define HANDLE_NULL(a) {if (a == NULL) { \
	printf( "Host memory failed in %s at line %d\n", __FILE__, __LINE__ ); \
    exit( EXIT_FAILURE );}}

#pragma endregion