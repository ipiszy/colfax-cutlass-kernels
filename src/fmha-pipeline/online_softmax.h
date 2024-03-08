#pragma once

#include "cute/arch/cluster_sm90.hpp"
#include "cutlass/bfloat16.h"
#include "cutlass/cutlass.h"
#include "cutlass/fast_math.h"
#include "cutlass/gemm/gemm.h"
#include "cutlass/layout/matrix.h"
#include "cutlass/layout/vector.h"
#include "cutlass/matrix.h"
#include "cutlass/numeric_types.h"
#include "cutlass/platform/platform.h"
#include "cutlass/tensor_ref.h"
#include <cute/tensor.hpp>
#include <cmath>

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct MaxOp {
  __device__ inline T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }
};

template <> struct MaxOp<float> {
  // This is slightly faster
  __device__ inline float operator()(float const &x, float const &y) {
    return max(x, y);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T> struct SumOp {
  __device__ inline T operator()(T const &x, T const &y) { return x + y; }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int THREADS> struct ShflReduce {
  static_assert(THREADS == 32 || THREADS == 16 || THREADS == 8 || THREADS == 4);
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator &op) {
    constexpr int OFFSET = THREADS / 2;
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, OFFSET));
    return ShflReduce<OFFSET>::run(x, op);
  }
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <> struct ShflReduce<2> {
  template <typename T, typename Operator>
  static __device__ inline T run(T x, Operator &op) {
    x = op(x, __shfl_xor_sync(uint32_t(-1), x, 1));
    return x;
  }
};

template <typename AccumType, typename Fragment0, typename Fragment1>
CUTLASS_DEVICE static void applySoftmaxNormalizer(const Fragment0 &sPrime,
                                                  Fragment1 &accum) {

  using FragValType = typename Fragment1::value_type;
  auto data = accum.data();
  int n = 0;
  int rowId = 0;
#pragma unroll
  for (int i = 0; i < size(shape<1>(accum)); ++i) {
    auto sPrime0 = AccumType(1.0 / sPrime(rowId));
    auto sPrime1 = AccumType(1.0 / sPrime(rowId + 1));

#pragma unroll
    for (int k = 0; k < size(shape<2>(accum)) * size<2>(shape<0>(accum)); ++k) {
      data[n] = FragValType(AccumType(data[n]) * sPrime0);
      n++;
      data[n] = FragValType(AccumType(data[n]) * sPrime0);
      n++;
      data[n] = FragValType(AccumType(data[n]) * sPrime1);
      n++;
      data[n] = FragValType(AccumType(data[n]) * sPrime1);
      n++;
    }
    rowId += 2;
  }
}

template <bool isFirst, typename AccumType, typename Fragment0,
          typename Fragment1, typename Fragment2, typename Fragment3,
          typename CountingTensor, typename GAccumShape>
CUTLASS_DEVICE static void
onlineSoftmaxAndRescale(Fragment0 &mi, Fragment1 &sPrime, Fragment2 &accum,
                        Fragment3 &accum_o, float scaleFactor,
                        const CountingTensor& countingTensor, const GAccumShape& gAccumShape,
                        int blockIdxY, int tileM, int tileN, int currentM, int currentK) {
  using namespace cute;
  using FragValType = typename Fragment2::value_type;
  using FragValTypeO = typename Fragment3::value_type;
  auto laneId = cutlass::canonical_lane_idx();

  // cute::axpby(scaleFactor, accum, Int<0>{}, accum);

  // First update `mi` to the max per-row
  //
  auto VT = shape<0>(accum); // number of vector elements per tile.
  auto MT = shape<1>(accum); // number of tiles along M.
  auto NT = shape<2>(accum); // number of tiles along N.
  if (cute::thread0()) {
    printf("VT: "); print(VT); printf("\n");
    printf("MT: "); print(MT); printf("\n");
    printf("NT: "); print(NT); printf("\n");
    printf("accum_o: "), print(accum_o); printf("\n");
  }
  MaxOp<AccumType> maxOp;

  auto M = get<0>(gAccumShape);
  auto N = get<1>(gAccumShape);

  auto data = accum.data();
  auto data_o = accum_o.data();
  int n = 0;
  int no = 0;
  Tensor miPrev = make_fragment_like(mi);
  cute::copy(mi, miPrev);

  int rowId = 0;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    auto max0 = mi(rowId);
    auto max1 = mi(rowId + 1);

    // Traverse 2-rows + 2-cols (2x2) simultaneously.

#pragma unroll
    for (int k = 0; k < NT * size<2>(VT); ++k) {
      auto coordinates = countingTensor(cute::make_tuple(0, 0, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < M) {
        data[n] = FragValType(AccumType(data[n]) * scaleFactor);
        // CUTE_LOG(
        //   "i=%d, M=%d, N=%d, k=%d, shape:((%d, %d, %d), %d, %d), x: %d, y: %d, n: %d, data[n]: %f, max0: %f\n",
        //   i, int(M), int(N), k, 0, 0, k%size<2>(VT), int(MT), int(NT), int(get<0>(coordinates)), int(get<1>(coordinates)), n, AccumType(data[n]), max0
        // );
        max0 = cutlass::fast_max(max0, AccumType(data[n]));
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0 && rowId == 0 &&
          (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2 || threadIdx.x == 3)
        ) {
          CUTE_LOG("calculate max, max0: %f, data[%d]: %f\n", float(max0), n, float(data[n]));
        }
      }
      n++;

      coordinates = countingTensor(cute::make_tuple(1, 0, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < currentM) {
        data[n] = FragValType(AccumType(data[n]) * scaleFactor);
        // CUTE_LOG(
        //   "i=%d, M=%d, N=%d, k=%d, shape:((%d, %d, %d), %d, %d), x: %d, y: %d, n: %d, data[n]: %f, max0: %f\n",
        //   i, int(M), int(N), k, 1, 0, k%size<2>(VT), int(MT), int(NT), int(get<0>(coordinates)), int(get<1>(coordinates)), n, AccumType(data[n]), max0
        // );
        max0 = cutlass::fast_max(max0, AccumType(data[n]));
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0 && rowId == 0 &&
          (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2 || threadIdx.x == 3)
        ) {
          CUTE_LOG("calculate max, max0: %f, data[%d]: %f\n", float(max0), n, float(data[n]));
        }
      }
      n++;

      coordinates = countingTensor(cute::make_tuple(0, 1, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < currentM) {
        data[n] = FragValType(AccumType(data[n]) * scaleFactor);
        max1 = cutlass::fast_max(max1, AccumType(data[n]));
      }
      n++;

      coordinates = countingTensor(cute::make_tuple(1, 1, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < currentM) {
        data[n] = FragValType(AccumType(data[n]) * scaleFactor);
        max1 = cutlass::fast_max(max1, AccumType(data[n]));
      }
      n++;
    }
    auto max_quad_0 = ShflReduce<4>::run(max0, maxOp);
    auto max_quad_1 = ShflReduce<4>::run(max1, maxOp);
    // CUTE_LOG("i=%d, MT=%d, NT=%d, max_quad_0: %f, max_quad_1: %f\n", i, int(MT), int(NT), max_quad_0, max_quad_1);
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0 && rowId == 0 &&
      (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2 || threadIdx.x == 3)
    ) {
      CUTE_LOG("calculate max reduction, max0: %f\n", float(max_quad_0));
    }
    mi(rowId) = max_quad_0;
    mi(rowId + 1) = max_quad_1;

    if (!isFirst) {
      float m_prime_exp0 = 0;
      float m_prime_exp1 = 0;
      auto mIdx = (threadIdx.x / 32) * 16 + (threadIdx.x % 32) / 4;
      auto nIdx = (threadIdx.x % 4) * 2;
      if (mIdx + blockIdx.x * tileM < currentM) {
        m_prime_exp0 = exp2f(miPrev(rowId) - max_quad_0);
        sPrime(rowId) *= m_prime_exp0;
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0 && rowId == 0 &&
          (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2 || threadIdx.x == 3)
        ) {
          CUTE_LOG("updated sPrime, m_prime_exp0: %f, result: %f\n", m_prime_exp0, sPrime(rowId));
        }
      }

      if (mIdx + 8 + blockIdx.x * tileM < currentM) {
        m_prime_exp1 = exp2f(miPrev(rowId + 1) - max_quad_1);
        sPrime(rowId + 1) *= m_prime_exp1;
      }

      for (int k = 0; k < size(shape<2>(accum_o)) * size<2>(shape<0>(accum_o));
           ++k) {
        if (mIdx + blockIdx.x * tileM < currentM && nIdx + k * 8 + blockIdxY * tileN < currentK) {
          data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp0);
          no++;
        }

        if (mIdx + blockIdx.x * tileM < currentM && nIdx + 1 + k * 8 + blockIdxY * tileN < currentK) {
          data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp0);
          no++;
        }

        if (mIdx + 8 + blockIdx.x * tileM < currentM && nIdx + k * 8 + blockIdxY * tileN < currentK) {
          data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp1);
          no++;
        }

        if (mIdx + 8 + blockIdx.x * tileM < currentM && nIdx + 1 + k * 8 + blockIdxY * tileN < currentK) {
          data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp1);
          no++;
        }
      }
    }
    rowId += 2;
  }

  SumOp<AccumType> sumOp;
  rowId = 0;
  n = 0;
#pragma unroll
  for (int i = 0; i < MT; ++i) {
    AccumType sum0 = 0.0f;
    AccumType sum1 = 0.0f;
    auto miRow0 = mi(rowId);
    auto miRow1 = mi(rowId + 1);
#pragma unroll
    for (int k = 0; k < NT * size<2>(VT); ++k) {

      auto coordinates = countingTensor(cute::make_tuple(0, 0, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < currentM) {
        auto val0 = AccumType(data[n]);
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0 && rowId == 0 &&
          (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2 || threadIdx.x == 3)
        ) {
          CUTE_LOG("calculate sum, n: %d, val: %f, miRow0: %f\n", int(n), float(val0), float(miRow0));
        }
        val0 = exp2f(val0 - miRow0);
        sum0 += val0;
        data[n] = val0;
      }
      n++;

      coordinates = countingTensor(cute::make_tuple(1, 0, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < currentM) {
        auto val1 = AccumType(data[n]);
        if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0 && rowId == 0 &&
          (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2 || threadIdx.x == 3)
        ) {
          CUTE_LOG("calculate sum, n: %d, val: %f, miRow0: %f\n", int(n), float(val1), float(miRow0));
        }
        val1 = exp2f(val1 - miRow0);
        sum0 += val1;
        data[n] = val1;
      }
      n++;

      coordinates = countingTensor(cute::make_tuple(0, 1, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < currentM) {
        auto val2 = AccumType(data[n]);
        val2 = exp2f(val2 - miRow1);
        sum1 += val2;
        data[n] = val2;
      }
      n++;

      coordinates = countingTensor(cute::make_tuple(1, 1, k % size<2>(VT)), MT, NT);
      if (get<0>(coordinates) + blockIdx.x * tileM < currentM && get<1>(coordinates) + blockIdxY * tileN < currentM) {
        auto val3 = AccumType(data[n]);
        val3 = exp2f(val3 - miRow1);
        sum1 += val3;
        data[n] = val3;
      }
      n++;
    }
    auto sumQuad0 = ShflReduce<4>::run(sum0, sumOp);
    auto sumQuad1 = ShflReduce<4>::run(sum1, sumOp);
    sPrime(rowId) += sumQuad0;
    sPrime(rowId + 1) += sumQuad1;
    if (blockIdx.x == 0 && blockIdx.y == 0 && blockIdx.z == 0 && threadIdx.y == 0 && threadIdx.z == 0 && rowId == 0 &&
      (threadIdx.x == 0 || threadIdx.x == 1 || threadIdx.x == 2 || threadIdx.x == 3)
    ) {
      CUTE_LOG("updated sPrime, sumQuad0: %f, result: %f\n", sumQuad0, sPrime(rowId));
    }
    rowId += 2;
  }
}
