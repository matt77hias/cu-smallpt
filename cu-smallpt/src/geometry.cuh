#pragma once

struct Ray {

	__device__ Ray(const Vector3 &o, const Vector3 &d, double tmin = 0.0, double tmax = INFINITY, int depth = 0)
		: o(o), d(d), tmin(tmin), tmax(tmax), depth(depth) {};

	__device__ inline Vector3 operator()(double t) const { return o + d * t; }

	Vector3 o, d;
	mutable double tmin, tmax;
	int depth;
};
