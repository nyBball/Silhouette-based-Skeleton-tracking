#ifndef UTILITY
#define UTILITY

#include <vector>
#include <string>
#include <math.h>
#include <algorithm>
#include <assert.h>
#include <Windows.h>

using namespace std;

namespace ohday
{

	const float ohday_pi = 3.141592653f;

	class Vector2f
	{
	public:
		float x, y;

		Vector2f()
		{
			x = y = 0.0f;
		}
		Vector2f(float fx, float fy)
		{
			x = fx;
			y = fy;
		}
		Vector2f(const Vector2f& v)
		{
			x = v.x;
			y = v.y;
		}

		Vector2f operator + (Vector2f v)
		{
			Vector2f v_res;
			v_res.x = v.x + x;
			v_res.y = v.y + y;
			return v_res;
		}

		Vector2f operator - (Vector2f v)
		{
			Vector2f v_res;
			v_res.x = x - v.x;
			v_res.y = y - v.y;
			return v_res;
		}

		Vector2f operator * (float k)
		{
			Vector2f v_res;
			v_res.x = k * x;
			v_res.y = k * y;
			return v_res;
		}

		float square()
		{
			return sqrt(x * x + y * y);
		}
	};

	class VectorNf
	{
	public:
		float exp;
		float len;
		bool is_exp_calced;
		bool is_len_calced;
		float dims;
		vector<float> data;

		VectorNf()
		{
			dims = 0;
			is_exp_calced = false;

			len = 0;
			is_len_calced = false;
		}

		VectorNf(int n)
		{
			is_exp_calced = false;
			is_len_calced = false;

			dims = n;
			data.resize(n);
			for (int i = 0; i < n; i++)
				data[i] = 0.0f;
		}

		VectorNf(const VectorNf& v)
		{
			dims = v.dims;
			data = v.data;

			exp = v.exp;
			is_exp_calced = v.is_exp_calced;

			len = v.len;
			is_len_calced = v.is_len_calced;
		}

		void resize(int n)
		{
			is_exp_calced = false;
			is_len_calced = false;

			dims = n;
			data.clear();
			data.resize(n);
			for (int i = 0; i < n; i++)
				data[i] = 0.0f;
		}

		float& operator [](int k)
		{
			assert(k >= 0 && k<dims);
			return data[k];
		}

		VectorNf operator -(VectorNf& v)
		{
			assert(v.dims == dims);
			VectorNf res(dims);
			for (int i = 0; i < dims; i++)
			{
				res[i] = data[i] - v[i];
			}
			return res;
		}

		VectorNf operator +(VectorNf& v)
		{
			assert(v.dims == dims);
			VectorNf res(dims);
			for (int i = 0; i < dims; i++)
			{
				res[i] = data[i] + v[i];
			}
			return res;
		}

		VectorNf operator *(float k)
		{
			VectorNf res(dims);
			for (int i = 0; i < dims; i++)
			{
				res[i] = data[i] * k;
			}
			return res;
		}

		float cross(VectorNf v)
		{
			assert(dims == v.dims);

			float sum = 0.0f;
			for (int i = 0; i < dims; i++)
			{
				sum += data[i] * v.data[i];
			}
			return sum;
		}

		void set(float *p_float)
		{
			for (int i = 0; i < dims; i++)
			{
				data[i] = p_float[i];
			}
		}

		float expectation()
		{
			if (is_exp_calced)
				return exp;

			float res = 0.0f;
			for (int i = 0; i < dims; i++)
			{
				res += data[i];
			}
			exp = res / dims;

			is_exp_calced = true;
			return exp;
		}

		float length()
		{
			if (is_len_calced)
				return len;


			float res = 0.0f;
			for (int i = 0; i < dims; i++)
			{
				res += data[i] * data[i];
			}
			len = sqrt(res);
			is_len_calced = true;
			return len;
		}

		void normalize()
		{
			float l = length();

			if (l != 0.0f)
			{
				for (int i = 0; i < dims; i++)
				{
					data[i] /= l;
				}
			}
		}

		VectorNf innermult(VectorNf& v)
		{
			VectorNf res(dims);
			for (int i = 0; i < dims; i++)
			{
				res[i] = data[i] * v[i];
			}
			return res;
		}

		float cov(VectorNf& v)//协方差
		{
			float xy_temp = 0.0f;
			for (int i = 0; i < dims; i++)
			{
				xy_temp += data[i] * v[i];
			}

			xy_temp /= dims;

			float res = xy_temp - this->expectation() * v.expectation();//cov=E(xy)-E(x)E(y)

			return res;
		}

		float variance()//方差 e(x^2)-e(x)^2
		{

			VectorNf x2 = this->innermult(*this);

			float ex = this->expectation();

			float res = x2.expectation() - ex * ex;

			return res;
		}


	};

	class RandomDevice
	{

	private:
		int gauss_phase;
		int seed;
	public:
		enum e_random_type
		{
			_linear = 0,
			_gaussian
		};
		RandomDevice()
		{
			gauss_phase = 0;
			ResetSeed();
		}

		void ResetSeed()
		{
			SYSTEMTIME sys;
			GetLocalTime(&sys);
			seed = sys.wMilliseconds;

			srand(seed);
		}

		float GetFloatLine(float f_min, float f_max)
		{
			int ra = rand() % RAND_MAX;
			float rand_res = f_min + (f_max - f_min) * float(rand()) / float(RAND_MAX);
			return rand_res;
		}

		float GetFloatGauss(float f_E, float f_S)
		{
			float u1, u2, v1, v2, s, x;
			do
			{
				u1 = (float)rand() / RAND_MAX;
				u2 = (float)rand() / RAND_MAX;

				v1 = 2 * u1 - 1;
				v2 = 2 * u2 - 1;

				s = v1 * v1 + v2 * v2;
			} while (s >= 1 || s == 0);

			if (!gauss_phase)
				x = v1 * sqrt(-2 * log(s) / s);
			else
				x = v2 * sqrt(-2 * log(s) / s);

			gauss_phase = 1 - gauss_phase;

			x = x * f_S + f_E;
			return x;
		}

		Vector2f GetVector2(float exp, float vari)
		{
			float theta = GetFloatLine(0, 2 * ohday_pi);
			float length = GetFloatGauss(exp, vari);

			Vector2f res;
			res.x = length * cos(theta);
			res.y = length * sin(theta);

			return res;
		}

		VectorNf GetVectorN(int n, float para_1, float para_2, e_random_type random_type = _linear)
		{
			VectorNf res(n);

			if (random_type == _linear)
			{
				for (int i = 0; i < n; i++)
				{
					res[i] = GetFloatLine(para_1, para_2);
				}
			}
			else
			{
				for (int i = 0; i < n; i++)
				{
					res[i] = GetFloatGauss(para_1, para_2);
				}
			}

			return res;
		}

	};
	class Delta
	{
	public:
		double value;
		int index;
		bool operator < (const Delta d)
		{
			return (value < d.value);
		}

	};
};

#endif //UTILITY