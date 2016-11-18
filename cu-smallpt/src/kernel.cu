
#include "cu-smallpt.h"

#include "specular.cuh"
#include "sphere.h"

// Scene
#define REFRACTIVE_INDEX_OUT 1.0
#define REFRACTIVE_INDEX_IN 1.5

//__constant__ Sphere dev_spheres[9];

Sphere spheres[] = {
	Sphere(1e5,  Vector3(1e5 + 1, 40.8, 81.6),   Vector3(),   Vector3(0.75, 0.25, 0.25), DIFFUSE), //Left
	Sphere(1e5,  Vector3(-1e5 + 99, 40.8, 81.6), Vector3(),   Vector3(0.25, 0.25, 0.75), DIFFUSE), //Right
	Sphere(1e5,  Vector3(50, 40.8, 1e5),		 Vector3(),   Vector3(0.75), DIFFUSE),			   //Back
	Sphere(1e5,  Vector3(50, 40.8, -1e5 + 170),  Vector3(),   Vector3(), DIFFUSE),				   //Front
	Sphere(1e5,  Vector3(50, 1e5, 81.6),		 Vector3(),   Vector3(0.75), DIFFUSE),			   //Bottom
	Sphere(1e5,  Vector3(50, -1e5 + 81.6, 81.6), Vector3(),   Vector3(0.75), DIFFUSE),			   //Top
	Sphere(16.5, Vector3(27, 16.5, 47),			 Vector3(),   Vector3(0.999), SPECULAR),	       //Mirror
	Sphere(16.5, Vector3(73, 16.5, 78),			 Vector3(),	  Vector3(0.999), REFRACTIVE),	       //Glass
	Sphere(600,	 Vector3(50, 681.6 - .27, 81.6), Vector3(12), Vector3(), DIFFUSE)				   //Light
};


__device__ bool Intersect(const Sphere *dev_spheres, size_t nb_spheres, const Ray &ray, size_t &id) {
	bool hit = false;
	for (size_t i = 0; i < nb_spheres; ++i) {
		if (dev_spheres[i].Intersect(ray)) {
			hit = true;
			id = i;
		}
	}
	return hit;
}


__device__ bool Intersect(const Sphere *dev_spheres, size_t nb_spheres, const Ray &ray) {
	for (size_t i = 0; i < nb_spheres; ++i)
		if (dev_spheres[i].Intersect(ray))
			return true;
	return false;
}


__device__ Vector3 Radiance(const Sphere *dev_spheres, size_t nb_spheres, const Ray &ray, curandState *state) {
	Ray r = ray;
	Vector3 L;
	Vector3 F(1.0);

	while (true) {
		size_t id;
		if (!Intersect(dev_spheres, nb_spheres, r, id))
			return L;

		const Sphere &shape = dev_spheres[id];
		const Vector3 p = r(r.tmax);
		const Vector3 n = (p - shape.p).Normalize();

		L += F * shape.e;
		F *= shape.f;

		// Russian roulette
		if (r.depth > 4) {
			const double continue_probability = shape.f.Max();
			if (curand_uniform_double(state) >= continue_probability)
				return L;
			F /= continue_probability;
		}

		// Next path segment
		switch (shape.reflection_t) {
		case SPECULAR: {
			const Vector3 d = IdealSpecularReflect(r.d, n);
			r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.depth + 1);
			break;
		}
		case REFRACTIVE: {
			double pr;
			const Vector3 d = IdealSpecularTransmit(r.d, n, REFRACTIVE_INDEX_OUT, REFRACTIVE_INDEX_IN, pr, state);
			F *= pr;
			r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.depth + 1);
			break;
		}
		default: {
			const Vector3 w = n.Dot(r.d) < 0 ? n : -n;
			const Vector3 u = ((abs(w.x) > 0.1 ? Vector3(0.0, 1.0, 0.0) : Vector3(1.0, 0.0, 0.0)).Cross(w)).Normalize();
			const Vector3 v = w.Cross(u);

			const Vector3 sample_d = CosineWeightedSampleOnHemisphere(curand_uniform_double(state), curand_uniform_double(state));
			const Vector3 d = (sample_d.x * u + sample_d.y * v + sample_d.z * w).Normalize();
			r = Ray(p, d, EPSILON_SPHERE, INFINITY, r.depth + 1);
		}
		}
	}
}


__global__ void kernel(const Sphere *dev_spheres, size_t nb_spheres, int w, int h, Vector3 *Ls, int nb_samples) {
	const int x = threadIdx.x + blockIdx.x * blockDim.x;
	const int y = threadIdx.y + blockIdx.y * blockDim.y;
	const int offset = x + y * blockDim.x * gridDim.x;

	if (x >= w || y >= h)
		return;

	// RNG
	curandState state;
	curand_init(offset, 0, 0, &state);

	const Vector3 eye = Vector3(50, 52, 295.6);
	const Vector3 gaze = Vector3(0, -0.042612, -1).Normalize();
	const double fov = 0.5135;
	const Vector3 cx = Vector3(w * fov / h, 0.0, 0.0);
	const Vector3 cy = (cx.Cross(gaze)).Normalize() * fov;

	for (int sy = 0, i = (h - 1 - y) * w + x; sy < 2; ++sy) // 2 subpixel row
		for (int sx = 0; sx < 2; ++sx) { // 2 subpixel column
			Vector3 L;
			for (int s = 0; s < nb_samples; s++) { // samples per subpixel
				const double u1 = 2.0 * curand_uniform_double(&state);
				const double u2 = 2.0 * curand_uniform_double(&state);
				const double dx = u1 < 1 ? sqrt(u1) - 1.0 : 1.0 - sqrt(2.0 - u1);
				const double dy = u2 < 1 ? sqrt(u2) - 1.0 : 1.0 - sqrt(2.0 - u2);
				Vector3 d = cx * (((sx + 0.5 + dx) / 2 + x) / w - 0.5) +
							cy * (((sy + 0.5 + dy) / 2 + y) / h - 0.5) + gaze;
				L += Radiance(dev_spheres, nb_spheres, Ray(eye + d * 130, d.Normalize(), EPSILON_SPHERE), &state) * (1.0 / nb_samples);
			}
			Ls[i] += 0.25 * Clamp(L);
		}
}


int main(int argc, char *argv[]) {
	const int nb_samples = (argc == 2) ? atoi(argv[1]) / 4 : 1;

	const int w = 1024;
	const int h = 768;
	const int nb_pixels = w * h;

	// Set up device memory
	//HANDLE_ERROR( cudaMemcpyToSymbol(dev_spheres, spheres, sizeof(spheres)) );
	Sphere *dev_spheres;
	const size_t nb_spheres = sizeof(spheres) / sizeof(Sphere);
	HANDLE_ERROR( cudaMalloc((void**)&dev_spheres, sizeof(spheres)) );
	HANDLE_ERROR( cudaMemcpy(dev_spheres, spheres, sizeof(spheres), cudaMemcpyHostToDevice) );
	Vector3 *dev_Ls;
	HANDLE_ERROR( cudaMalloc((void**)&dev_Ls, nb_pixels * sizeof(Vector3)) );
	HANDLE_ERROR( cudaMemset(dev_Ls, 0, nb_pixels * sizeof(Vector3)) );
	
	// Kernel execution
	const dim3 nblocks(w / 16, h / 16);
	const dim3 nthreads(16, 16);
	kernel << <nblocks, nthreads >> >(dev_spheres, nb_spheres, w, h, dev_Ls, nb_samples);

	// Set up host memory
	Vector3 *Ls = (Vector3 *)malloc(nb_pixels * sizeof(Vector3));
	// Transfer device -> host
	HANDLE_ERROR( cudaMemcpy(Ls, dev_Ls, nb_pixels * sizeof(Vector3), cudaMemcpyDeviceToHost) );

	// Clean up device memory
	HANDLE_ERROR( cudaFree(dev_Ls) );
	HANDLE_ERROR( cudaFree(dev_spheres) );

	WritePPM(w, h, Ls);

	// Clean up host memory
	free(Ls);
}