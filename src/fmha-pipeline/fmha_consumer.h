#pragma once

#include "online_softmax.h"
#include "reg2reg.h"
#include "shared_storage.h"

// FMHA Consumer does GEMMs and softmax
template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class TiledMma0, class TiledMma1, class TileShapeS, class GmemLayoutS,
          typename TensorQ, typename TensorK, typename TensorS,
          typename TensorV, typename TensorO, typename RegLayout, typename Reg2Reg,
          typename RowMax, typename RowSum, bool UseVarSeqLen>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardConsumer(Gemm1Type const *Q, Gemm1Type const *K, Gemm2Type const *V,
                    Gemm1Type *S, const TensorQ &tSrQ, const TensorK &tSrK,
                    TensorS tSrS, const TensorV &tOrV, TensorO &tOrO,
                    const RegLayout &tOrPLayout, Reg2Reg & reg2reg, RowMax &rowMax, RowSum &rowSum,
                    const TileShapeS &tileShapeS,
                    const GmemLayoutS &gmemLayoutS, float scale, int blockIdxY,
                    const TiledMma0 &tiledMma0, const TiledMma1 &tiledMma1,
                    const AccumType &, const SoftType &, int m, int k) {
  using namespace cute;

  clear(tSrS);

  // Issue GEMM-I.
  cfk::gemm(tiledMma0, tSrQ, tSrK, tSrS);

  // Get the block coordinates for this CTA.
  auto blockIdxX = uint64_t(blockIdx.x);
  auto blockIdxH = uint64_t(blockIdx.y);
  auto blockIdxB = uint64_t(blockIdx.z);
  auto blkCoordS = make_coord(blockIdxX, blockIdxY, blockIdxH, blockIdxB);
  auto threadMma0 = tiledMma0.get_thread_slice(threadIdx.x);
  Tensor mS = make_tensor(make_gmem_ptr(S), gmemLayoutS);
  Tensor gS = local_tile(mS, tileShapeS, blkCoordS);
  Tensor tSgS = threadMma0.partition_C(gS);
  Tensor gSCounting = make_identity_tensor(gS.shape());
  Tensor tSgSCounting = threadMma0.partition_C(gSCounting);
  Tensor tSpS = make_tensor<bool>(get<0>(tSgSCounting.shape()));
  static_assert(rank(tSgSCounting) == 3);
  static_assert(get<1>(tSgSCounting.shape()) == 1);
  static_assert(get<2>(tSgSCounting.shape()) == 1);

#ifdef COPYOUTMM0
  auto VT = shape<0>(tSgS); // number of vector elements per tile.
  auto MT = shape<1>(tSgS); // number of tiles along M.
  auto NT = shape<2>(tSgS); // number of tiles along N.
  static_assert(get<0>(VT) == 2);
  static_assert(get<1>(VT) == 2);

  fillSPredicate(
    tSgSCounting, tSpS, get<0>(tileShapeS),
    get<1>(tileShapeS), blockIdx.x, blockIdxY, m
  );
  copy_if(tSpS, tSrS, tSgS);
  cute::cp_async_wait<0>();
  cutlass::arch::NamedBarrier::sync(size(TiledMma0{}), 0);
#endif

  if (!UseVarSeqLen || isWarpInBound(blockIdxY, get<0>(tileShapeS), get<1>(tileShapeS), m)) {
    // Fast path. No need to check if all elements in a warp are in bound again.
    if (blockIdxY == 0) { // Compute Online Softmax and NO Output Rescaling.
      onlineSoftmaxAndRescaleAllInBound<true, SoftType>(
        rowMax, rowSum, tSrS, tOrO, scale,
        blockIdxY, get<0>(tileShapeS), get<1>(tileShapeS), m, k
      );
    } else { // Compute Online Softmax and Output Rescaling.
      onlineSoftmaxAndRescaleAllInBound<false, SoftType>(
        rowMax, rowSum, tSrS, tOrO, scale,
        blockIdxY, get<0>(tileShapeS), get<1>(tileShapeS), m, k
      );
    }
  } else {
#ifndef COPYOUTMM0
    fillSPredicate(
      tSgSCounting, tSpS, get<0>(tileShapeS),
      get<1>(tileShapeS), blockIdx.x, blockIdxY, m
    );
#endif
    // Slow path. Need to check if all elements in a warp are in bound.
    if (blockIdxY == 0) { // Compute Online Softmax and NO Output Rescaling.
      onlineSoftmaxAndRescale<true, SoftType>(
        rowMax, rowSum, tSrS, tOrO, scale,
        blockIdxY, get<0>(tileShapeS), get<1>(tileShapeS), m, k, tSpS
      );
    } else { // Compute Online Softmax and Output Rescaling.
      onlineSoftmaxAndRescale<false, SoftType>(
        rowMax, rowSum, tSrS, tOrO, scale,
        blockIdxY, get<0>(tileShapeS), get<1>(tileShapeS), m, k, tSpS
      );
    }
  }
  warpgroup_fence_operand(tSrS);

  // ISSUE GEMM-II with Operand A from RMEM.
  // Convert Operand A from SoftType [=float or half] to Gemm2Type [=half_t or
  // fp8] before issuing.
  auto tSrSPrec = convert_type<Gemm2Type, AccumType>(tSrS);
  // Invoke additional register permute/shuffle if GEMM-II is FP8.
#ifdef GEMM2FP8
  reg2reg(tSrSPrec);
#endif
  auto tOrP = make_tensor(tSrSPrec.data(), tOrPLayout);
  warpgroup_fence_operand(tSrS);
  // Issue GEMM-II.
  cfk::gemm(tiledMma1, tOrP, tOrV, tOrO);
}
