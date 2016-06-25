/*

Copyright 2016 Thomas Luu

   Licensed under the Apache License, Version 2.0 (the "License");
   you may not use this file except in compliance with the License.
   You may obtain a copy of the License at

       http://www.apache.org/licenses/LICENSE-2.0

   Unless required by applicable law or agreed to in writing, software
   distributed under the License is distributed on an "AS IS" BASIS,
   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
   See the License for the specific language governing permissions and
   limitations under the License.

*/

/*

File: ncx2cdfinv.cu

Computation of the noncentral chi-squared quantile function.

Based on:

Luu, T; (2016) Fast and accurate parallel computation of quantile functions for 
random number generation. Doctoral thesis, UCL (University College London).

http://discovery.ucl.ac.uk/1482128/

*/

#ifndef NCX2CDFINV
#define NCX2CDFINV

#include <math_constants.h>

#if 1
#define TOL 0.1
#else
#define TOL 0.01
#endif

__host__ __device__ inline double sankaran(double u, double k, double l)
{
  double h = 1.0 - CUDART_TWOTHIRD * (k + l) * (k + l + l + l) / ((k + l + l) * (k + l + l));
  double p = (k + l + l) / ((k + l) * (k + l));
  double m = (h - 1.0) * (1.0 - (h + h + h));
  double mu = 1.0 + h * p * (h - 1.0 - (1.0 - h * 0.5) * m * p);
  double s = h * sqrt(p + p) * (1 + m * p * 0.5);
  double z = normcdfinv(u);
  double x = z * s + mu;
  return (k + l) * pow(x, 1.0 / h);
}

__host__ __device__ inline double v(double u, double k, double l, double c)
{
  return pow(c * u, 2.0 / k);
}

__host__ __device__ inline double v_inv(double x, double k, double l, double c)
{
  return pow(x, 0.5 * k) / c;
}

__host__ __device__ inline double luu(double u, double k, double l, double *u_split = 0)
{
  double c = 0.5 * exp2(0.5 * k) * exp(0.5 * l) * k * tgamma(0.5 * k);
  double vv = v(u, k, l, c);

  double v_split;
  if (u_split != 0) {
    double rg2 = (k * (2.0 + k)) / (k - l);
    double rg3 = (2.0 * k * k * (2.0 + k) * (2.0 + k) * (4.0 + k)) / (6.0 * l * l * (k - 1) + 2.0 * l * (8.0 - 5.0 * k) * k + k * k * (5.0 * k - 8.0));
    if (k == l) {
      v_split = cbrt(TOL * fabs(rg3));
    } else {
      v_split = sqrt(TOL * fabs(rg2));
    }
    *u_split = v_inv(v_split, k, l, c);
  }

  return vv;
}

__host__ __device__ inline double ncx2cdfinv(double u, double k, double l)
{
  if (u == 0.0) {
    return 0.0;
  }

  if (u == 1.0) {
#ifdef __CUDA_ARCH__
    return CUDART_INF;
#else
    return INFINITY;
#endif
  }

  double sankaran_approx = sankaran(u, k, l);

  double u_split;
  double luu_approx = luu(u, k, l, &u_split);

  if (isnan(sankaran_approx)) return luu_approx;
  return u < u_split ? luu_approx : sankaran_approx;
}

#endif

