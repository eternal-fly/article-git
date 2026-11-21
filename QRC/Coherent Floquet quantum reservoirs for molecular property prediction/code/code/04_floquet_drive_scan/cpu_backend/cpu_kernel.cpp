// Exact pure-suffix evolution; no stochastic trajectories or suffix truncation.
#include <algorithm>
#include <cmath>
#include <cstddef>
#include <vector>
#include <omp.h>

static inline void gate(float* r, float* im, const float* u, int dim, int stride) {
    const float ar=u[0], ai=u[1], br=u[2], bi=u[3];
    const float cr=u[4], ci=u[5], dr=u[6], di=u[7];
    for (int base=0; base<dim; base+=2*stride) {
        #pragma omp simd
        for (int j=0; j<stride; ++j) {
            const int a=base+j, b=a+stride;
            const float xr=r[a], xi=im[a], yr=r[b], yi=im[b];
            r[a]=ar*xr-ai*xi+br*yr-bi*yi;
            im[a]=ar*xi+ai*xr+br*yi+bi*yr;
            r[b]=cr*xr-ci*xi+dr*yr-di*yi;
            im[b]=cr*xi+ci*xr+dr*yi+di*yr;
        }
    }
}
static inline void rx(float* r, float* im, int dim, int stride, float c, float s) {
    for (int base=0; base<dim; base+=2*stride) {
        #pragma omp simd
        for (int j=0; j<stride; ++j) {
            const int a=base+j, b=a+stride;
            const float xr=r[a], xi=im[a], yr=r[b], yi=im[b];
            r[a]=c*xr+s*yi; im[a]=c*xi-s*yr;
            r[b]=c*yr+s*xi; im[b]=c*yi-s*xr;
        }
    }
}
static inline void diagonal(float* r, float* im, const float* phase, int dim) {
    #pragma omp simd
    for (int i=0; i<dim; ++i) {
        const float xr=r[i], xi=im[i], pr=phase[2*i], pi=phase[2*i+1];
        r[i]=xr*pr-xi*pi; im[i]=xr*pi+xi*pr;
    }
}
static inline void cycle(float* r, float* im, const float* phase, int dim, int q, float c, float s) {
    for (int wire=0; wire<q; ++wire) rx(r, im, dim, 1<<(q-1-wire), c, s);
    diagonal(r, im, phase, dim);
}
extern "C" int hopping_suffixes(const float* gates, const float* phase,
    float cosine, float sine, int n, int t, int q, int layers, int threads,
    double* output, double* errors) {
    if (n<1 || t<1 || q<1 || q>16 || layers<1 || threads<1) return -1;
    const int dim=1<<q, width=q*(q+1)/2;
    std::vector<int> masks;
    for (int i=0; i<q; ++i) masks.push_back(1<<(q-1-i));
    for (int i=0; i<q; ++i) for (int j=i+1; j<q; ++j)
        masks.push_back((1<<(q-1-i)) ^ (1<<(q-1-j)));
    #pragma omp parallel num_threads(threads)
    {
        std::vector<float> real(dim), imag(dim);
        std::vector<double> walsh(dim);
        #pragma omp for schedule(dynamic, 1)
        for (int row=0; row<n; ++row) {
            double max_error=0;
            for (int suffix=0; suffix<t; ++suffix) {
                std::fill(real.begin(), real.end(), 0.f);
                std::fill(imag.begin(), imag.end(), 0.f);
                real[0]=1.f;
                for (int event=suffix; event<t; ++event) {
                    // The first RX commutes with other wires' input gates.
                    // Fuse it only with its own wire's Hamiltonian input gate.
                    for (int wire=0; wire<q; ++wire)
                        gate(real.data(), imag.data(), gates+((std::size_t(row)*t+event)*q+wire)*8,
                             dim, 1<<(q-1-wire));
                    diagonal(real.data(), imag.data(), phase, dim);
                    for (int repeat=1; repeat<layers; ++repeat)
                        cycle(real.data(), imag.data(), phase, dim, q, cosine, sine);
                }
                for (int snapshot=0; snapshot<3; ++snapshot) {
                    if (snapshot) cycle(real.data(), imag.data(), phase, dim, q, cosine, sine);
                    for (int i=0; i<dim; ++i)
                        walsh[i]=double(real[i])*real[i]+double(imag[i])*imag[i];
                    for (int stride=1; stride<dim; stride*=2) for (int base=0; base<dim; base+=2*stride) {
                        #pragma omp simd
                        for (int j=0; j<stride; ++j) {
                            const double a=walsh[base+j], b=walsh[base+j+stride];
                            walsh[base+j]=a+b; walsh[base+j+stride]=a-b;
                        }
                    }
                    if (!std::isfinite(walsh[0])) max_error=INFINITY;
                    else max_error=std::max(max_error, std::abs(walsh[0]-1.));
                    double* dest=output+((std::size_t(suffix)*n+row)*3+snapshot)*width;
                    for (int j=0; j<width; ++j) dest[j]=walsh[masks[j]];
                }
            }
            errors[row]=max_error;
        }
    }
    return 0;
}
