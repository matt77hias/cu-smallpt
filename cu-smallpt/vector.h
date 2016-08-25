#pragma once

struct Vector3 {

	union {
		struct {
			double x, y, z;
		};
		double raw[3];
	};

	__host__ __device__ Vector3(double a = 0) : x(a), y(a), z(a) {}
	__host__ __device__ Vector3(double x, double y, double z) : x(x), y(y), z(z) {}
	__host__ __device__ Vector3(const Vector3 &v) : x(v.x), y(v.y), z(v.z) {}

	__device__ inline bool HasNaNs() const {
		return isnan(x) || isnan(y) || isnan(z);
	}

	__device__ inline Vector3 &operator=(const Vector3 &v) {
		x = v.x;
		y = v.y;
		z = v.z;
		return *this;
	}

	__device__ inline Vector3 operator-() const {
		return Vector3(-x, -y, -z);
	}
	__device__ inline Vector3 operator+(const Vector3 &v) const {
		return Vector3(x + v.x, y + v.y, z + v.z);
	}
	__device__ inline Vector3 operator-(const Vector3 &v) const {
		return Vector3(x - v.x, y - v.y, z - v.z);
	}
	__device__ inline Vector3 operator*(const Vector3 &v) const {
		return Vector3(x * v.x, y * v.y, z * v.z);
	}
	__device__ inline Vector3 operator/(const Vector3 &v) const {
		return Vector3(x / v.x, y / v.y, z / v.z);
	}
	__device__ inline Vector3 operator+(double a) const {
		return Vector3(x + a, y + a, z + a);
	}
	__device__ inline Vector3 operator-(double a) const {
		return Vector3(x - a, y - a, z - a);
	}
	__device__ inline Vector3 operator*(double a) const {
		return Vector3(x * a, y * a, z * a);
	}
	__device__ inline Vector3 operator/(double a) const {
		const double inv_a = 1.0 / a;
		return Vector3(x * inv_a, y * inv_a, z * inv_a);
	}
	__device__ friend inline Vector3 operator+(double a, const Vector3 &v) {
		return Vector3(v.x + a, v.y + a, v.z + a);
	}
	__device__ friend inline Vector3 operator-(double a, const Vector3 &v) {
		return Vector3(v.x - a, v.y - a, v.z - a);
	}
	__device__ friend inline Vector3 operator*(double a, const Vector3 &v) {
		return Vector3(v.x * a, v.y * a, v.z * a);
	}
	__device__ friend inline Vector3 operator/(double a, const Vector3 &v) {
		const double inv_a = 1.0 / a;
		return Vector3(v.x * inv_a, v.y * inv_a, v.z * inv_a);
	}

	__device__ inline Vector3 &operator+=(const Vector3 &v) {
		x += v.x;
		y += v.y;
		z += v.z;
		return *this;
	}
	__device__ inline Vector3 &operator-=(const Vector3 &v) {
		x -= v.x;
		y -= v.y;
		z -= v.z;
		return *this;
	}
	__device__ inline Vector3 &operator*=(const Vector3 &v) {
		x *= v.x;
		y *= v.y;
		z *= v.z;
		return *this;
	}
	__device__ inline Vector3 &operator/=(const Vector3 &v) {
		x /= v.x;
		y /= v.y;
		z /= v.z;
		return *this;
	}
	__device__ inline Vector3 &operator+=(double a) {
		x += a;
		y += a;
		z += a;
		return *this;
	}
	__device__ inline Vector3 &operator-=(double a) {
		x -= a;
		y -= a;
		z -= a;
		return *this;
	}
	__device__ inline Vector3 &operator*=(double a) {
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}
	__device__ inline Vector3 &operator/=(double a) {
		const double inv_a = 1.0 / a;
		x *= inv_a;
		y *= inv_a;
		z *= inv_a;
		return *this;
	}

	__device__ inline double Dot(const Vector3 &v) const {
		return x * v.x + y * v.y + z * v.z;
	}
	__device__ inline Vector3 Cross(const Vector3 &v) const {
		return Vector3(y * v.z - z * v.y, z * v.x - x * v.z, x * v.y - y * v.x);
	}

	__device__ inline bool operator==(const Vector3 &v) const {
		return x == v.x && y == v.y && z == v.z;
	}
	__device__ inline bool operator!=(const Vector3 &v) const {
		return x != v.x || y != v.y || z != v.z;
	}
	__device__ inline bool operator<(const Vector3 &v) const {
		return x < v.x && y < v.y && z < v.z;
	}
	__device__ inline bool operator<=(const Vector3 &v) const {
		return x <= v.x && y <= v.y && z <= v.z;
	}
	__device__ inline bool operator>(const Vector3 &v) const {
		return x > v.x && y > v.y && z > v.z;
	}
	__device__ inline bool operator>=(const Vector3 &v) const {
		return x >= v.x && y >= v.y && z >= v.z;
	}

	__device__ friend inline Vector3 Sqrt(const Vector3 &v) {
		return Vector3(sqrt(v.x), sqrt(v.y), sqrt(v.z));
	}
	__device__ friend inline Vector3 Pow(const Vector3 &v, double a) {
		return Vector3(pow(v.x, a), pow(v.y, a), pow(v.z, a));
	}
	__device__ friend inline Vector3 Abs(const Vector3 &v) {
		return Vector3(abs(v.x), abs(v.y), abs(v.z));
	}
	__device__ friend inline Vector3 Min(const Vector3 &v1, const Vector3 &v2) {
		return Vector3(fmin(v1.x, v2.x), fmin(v1.y, v2.y), fmin(v1.z, v2.z));
	}
	__device__ friend inline Vector3 Max(const Vector3 &v1, const Vector3 &v2) {
		return Vector3(fmax(v1.x, v2.x), fmax(v1.y, v2.y), fmax(v1.z, v2.z));
	}
	__device__ friend inline Vector3 Round(const Vector3 &v) {
		return Vector3(round(v.x), round(v.y), round(v.z));
	}
	__device__ friend inline Vector3 Floor(const Vector3 &v) {
		return Vector3(floor(v.x), floor(v.y), floor(v.z));
	}
	__device__ friend inline Vector3 Ceil(const Vector3 &v) {
		return Vector3(ceil(v.x), ceil(v.y), ceil(v.z));
	}
	__device__ friend inline Vector3 Trunc(const Vector3 &v) {
		return Vector3(trunc(v.x), trunc(v.y), trunc(v.z));
	}
	__device__ friend inline Vector3 Clamp(const Vector3 &v, double low = 0, double high = 1) {
		return Vector3(Clamp(v.x, low, high), Clamp(v.y, low, high), Clamp(v.z, low, high));
	}
	__device__ friend inline Vector3 Lerp(double a, const Vector3 &v1, const Vector3 &v2) {
		return v1 + a * (v2 - v1);
	}
	__device__ friend inline Vector3 Permute(const Vector3 &v, int x, int y, int z) {
		return Vector3(v[x], v[y], v[z]);
	}

	__device__ inline double operator[](size_t i) const {
		return raw[i];
	}
	__device__ inline double &operator[](size_t i) {
		return raw[i];
	}

	__device__ inline int MinDimension() const {
		return (x < y && x < z) ? 0 : ((y < z) ? 1 : 2);
	}
	__device__ inline int MaxDimension() const {
		return (x > y && x > z) ? 0 : ((y > z) ? 1 : 2);
	}
	__device__ inline double Min() const {
		return (x < y && x < z) ? x : ((y < z) ? y : z);
	}
	__device__ inline double Max() const {
		return (x > y && x > z) ? x : ((y > z) ? y : z);
	}

	__device__ inline double Norm2_squared() const {
		return x * x + y * y + z * z;
	}
	__device__ inline double Norm2() const {
		return sqrt(Norm2_squared());
	}
	__device__ inline Vector3 &Normalize() {
		const double a = 1 / Norm2();
		x *= a;
		y *= a;
		z *= a;
		return *this;
	}
};