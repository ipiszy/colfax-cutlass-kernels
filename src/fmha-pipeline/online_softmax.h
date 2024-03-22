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

__device__
inline bool isIdxInBound(int idx, int blockIdx, int tileSize, int total) {
  return idx + blockIdx * tileSize < total;
}

__device__
inline bool isWarpInBound(int blockIdxY, int tileShapeX, int tileShapeY, int m) {
  return
    ((threadIdx.x / 32) * 16 + blockIdx.x * tileShapeX <= m) &&
    ((blockIdxY + 1) * tileShapeY <= m);
}

template <typename SCountingTensor, typename SPTensor>
__device__ inline void fillSPredicate(
    const SCountingTensor& tensorS, SPTensor& tensorP,
    int tileX, int tileY, int blockIdxX, int blockIdxY, int N) {
#pragma unroll
  for (int i = 0; i < get<0,0>(tensorS.shape()); ++i) {
#pragma unroll
    for (int j = 0; j < get<0,1>(tensorS.shape()); ++j) {
#pragma unroll
      for (int k = 0; k < get<0,2>(tensorS.shape()); ++k) {
        auto sCoordinates = tensorS(cute::make_tuple(i,j,k),0,0);
        tensorP(i, j, k) = (
          get<0>(sCoordinates) + tileX * blockIdxX < N
        ) && (
          get<1>(sCoordinates) + tileY * blockIdxY < N
        );
      }
    }
  }
}

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
          typename TensorP>
CUTLASS_DEVICE static void
onlineSoftmaxAndRescale(Fragment0 &mi, Fragment1 &sPrime, Fragment2 &accum,
                        Fragment3 &accum_o, float scaleFactor,
                        int blockIdxY, int tileM, int tileN, int currentM, int currentK,
                        const TensorP& tensorP) {
  using namespace cute;
  using FragValType = typename Fragment2::value_type;
  using FragValTypeO = typename Fragment3::value_type;

  // cute::axpby(scaleFactor, accum, Int<0>{}, accum);

  // First update `mi` to the max per-row
  MaxOp<AccumType> maxOp;

  Tensor miPrev = make_fragment_like(mi);
  cute::copy(mi, miPrev);

#pragma unroll
  for (auto rowId = 0; rowId < get<1>(tensorP.shape()); ++rowId) {
    auto maxElem = mi(rowId);
#pragma unroll
    for (auto i = 0; i < get<0>(tensorP.shape()); ++i) {
#pragma unroll
      for (auto k = 0; k < get<2>(tensorP.shape()); ++k) {
        if (tensorP(i, rowId, k)) {
          accum(cute::make_tuple(i, rowId, k), 0, 0) = FragValType(
            AccumType(accum(cute::make_tuple(i, rowId, k), 0, 0)) * scaleFactor
          );
          maxElem = cutlass::fast_max(
            maxElem,
            AccumType(accum(cute::make_tuple(i, rowId, k), 0, 0))
          );
        }
      }
    }
    auto maxQuad = ShflReduce<4>::run(maxElem, maxOp);
    mi(rowId) = maxQuad;
  }

  if (!isFirst) {
#pragma unroll
    for (int rowId = 0; rowId < get<0, 1>(accum_o.shape()); ++rowId) {
      float m_prime_exp = exp2f(miPrev(rowId) - mi(rowId));
      sPrime(rowId) *= m_prime_exp;
#pragma unroll
      for (int i = 0; i < get<0, 0>(accum_o.shape()); ++i) {
#pragma unroll
        for (int k = 0; k < get<0, 2>(accum_o.shape()); ++k) {
          accum_o(cute::make_tuple(i, rowId, k), 0, 0) =
            FragValTypeO(AccumType(accum_o(cute::make_tuple(i, rowId, k), 0, 0)) * m_prime_exp);
        }
      }
    }
  }

  SumOp<AccumType> sumOp;

#pragma unroll
  for (int rowId = 0; rowId < get<1>(tensorP.shape()); ++rowId) {
    AccumType sumElem = 0.0f;
    auto miRow = mi(rowId);
#pragma unroll
    for (int k = 0; k < get<2>(tensorP.shape()); ++k) {
#pragma unroll
      for (int i = 0; i < get<0>(tensorP.shape()); ++i) {
        if (tensorP(i, rowId, k)) {
          auto val = AccumType(accum(cute::make_tuple(i, rowId, k), 0, 0));
          val = exp2f(val - miRow);
          sumElem += val;
          accum(cute::make_tuple(i, rowId, k), 0, 0) = val;
       }
      }
    }
    auto sumQuad = ShflReduce<4>::run(sumElem, sumOp);
    sPrime(rowId) += sumQuad;
  }
}

template <bool isFirst, typename AccumType, typename Fragment0,
          typename Fragment1, typename Fragment2, typename Fragment3>
CUTLASS_DEVICE static void
onlineSoftmaxAndRescaleAllInBound(
    Fragment0 &mi, Fragment1 &sPrime, Fragment2 &accum,
    Fragment3 &accum_o, float scaleFactor,
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
  static_assert(get<0>(VT) == 2);
  static_assert(get<1>(VT) == 2);
  static_assert(MT == 1);
  static_assert(NT == 1);

  MaxOp<AccumType> maxOp;

  auto data = accum.data();
  auto data_o = accum_o.data();
  int n = 0;
  int no = 0;
  Tensor miPrev = make_fragment_like(mi);
  cute::copy(mi, miPrev);

  int rowId = 0;
  auto max0 = mi(rowId);
  auto max1 = mi(rowId + 1);

  // Traverse 2-rows + 2-cols (2x2) simultaneously.

#pragma unroll
  for (int k = 0; k < NT * size<2>(VT); ++k) {
    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max0 = cutlass::fast_max(max0, AccumType(data[n]));
    n++;

    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max0 = cutlass::fast_max(max0, AccumType(data[n]));
    n++;

    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max1 = cutlass::fast_max(max1, AccumType(data[n]));
    n++;

    data[n] = FragValType(AccumType(data[n]) * scaleFactor);
    max1 = cutlass::fast_max(max1, AccumType(data[n]));
    n++;
  }
  auto max_quad_0 = ShflReduce<4>::run(max0, maxOp);
  auto max_quad_1 = ShflReduce<4>::run(max1, maxOp);

  mi(rowId) = max_quad_0;
  mi(rowId + 1) = max_quad_1;

  if (!isFirst) {
    float m_prime_exp0 = exp2f(miPrev(rowId) - max_quad_0);
    sPrime(rowId) *= m_prime_exp0;

    float m_prime_exp1 = exp2f(miPrev(rowId + 1) - max_quad_1);
    sPrime(rowId + 1) *= m_prime_exp1;

    for (int k = 0; k < size(shape<2>(accum_o)) * size<2>(shape<0>(accum_o));
         ++k) {
      data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp0);
      no++;

      data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp0);
      no++;

      data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp1);
      no++;

      data_o[no] = FragValTypeO(AccumType(data_o[no]) * m_prime_exp1);
      no++;
    }
  }

  SumOp<AccumType> sumOp;
  rowId = 0;
  n = 0;

  AccumType sum0 = 0.0f;
  AccumType sum1 = 0.0f;
  auto miRow0 = mi(rowId);
  auto miRow1 = mi(rowId + 1);

#pragma unroll
  for (int k = 0; k < NT * size<2>(VT); ++k) {
    auto val0 = AccumType(data[n]);
    val0 = exp2f(val0 - miRow0);
    sum0 += val0;
    data[n] = val0;
    n++;

    auto val1 = AccumType(data[n]);
    val1 = exp2f(val1 - miRow0);
    sum0 += val1;
    data[n] = val1;
    n++;

    auto val2 = AccumType(data[n]);
    val2 = exp2f(val2 - miRow1);
    sum1 += val2;
    data[n] = val2;
    n++;

    auto val3 = AccumType(data[n]);
    val3 = exp2f(val3 - miRow1);
    sum1 += val3;
    data[n] = val3;
    n++;
  }
  auto sumQuad0 = ShflReduce<4>::run(sum0, sumOp);
  auto sumQuad1 = ShflReduce<4>::run(sum1, sumOp);
  sPrime(rowId) += sumQuad0;
  sPrime(rowId + 1) += sumQuad1;
}
