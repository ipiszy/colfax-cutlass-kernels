#pragma once

#include "online_softmax.h"
#include "reg2reg.h"
#include "shared_storage.h"

// FMHA Consumer does GEMMs and softmax
template <class Gemm1Type, class AccumType, class SoftType, class Gemm2Type,
          class TiledMma0, class TiledMma1, class TileShapeS, class GmemLayoutS,
          typename TensorQ, typename TensorK, typename TensorS,
          typename TensorV, typename TensorO, typename RegLayout, typename Reg2Reg,
          typename RowMax, typename RowSum>
__device__ static void //__launch_bounds__(128, 2)
fmhaForwardConsumer(Gemm1Type const *Q, Gemm1Type const *K, Gemm2Type const *V,
                    Gemm1Type *S, const TensorQ &tSrQ, const TensorK &tSrK,
                    TensorS &&tSrS, const TensorV &tOrV, TensorO &tOrO,
                    const RegLayout &tOrPLayout, Reg2Reg & reg2reg, RowMax &rowMax, RowSum &rowSum,
                    const TileShapeS &tileShapeS,
                    const GmemLayoutS &gmemLayoutS, float scale, int blockIdxY,
                    const TiledMma0 &tiledMma0, const TiledMma1 &tiledMma1,
                    const AccumType &, const SoftType &, int* thread_count) {

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
Tensor gSCounting = make_identity_tensor(gS.shape());
Tensor tSgS = threadMma0.partition_C(gS);
Tensor tSgSCounting = threadMma0.partition_C(gSCounting);

// Required for verification ONLY.
#ifdef COPYOUTMM0
  // if (cute::thread0()) {
  //   printf("tSrS: "); print(tSrS); print("\n");
  //   printf("tSgS: "); print(tSgS); print("\n");
  // }
  for (int i = 0; i < get<0,0>(tSgSCounting.shape()); ++i) {
    for (int j = 0; j < get<0,1>(tSgSCounting.shape()); ++j) {
      for (int k = 0; k < get<0,2>(tSgSCounting.shape()); ++k) {
        auto sCoordinates = tSgSCounting(cute::make_tuple(i,j,k),0,0);
        if (get<0>(sCoordinates) < get<0>(gmemLayoutS.shape()) && get<1>(sCoordinates) < get<1>(gmemLayoutS.shape())) {
          copy(tSrS(cute::make_tuple(i,j,k),_,_), tSgS(cute::make_tuple(i,j,k),_,_));
        }
        // if (cute::thread0()) {
        //   printf("tSgS_shape(%d,%d,%d):  ", i, j, k); print(tSgSCounting(cute::make_tuple(i,j,k),0,0)); print("\n");
        // }
      }
    }
  }
  cute::cp_async_wait<0>();
  cutlass::arch::NamedBarrier::sync(size(TiledMma0{}), 0);
#endif

  if (blockIdxY == 0) { // Compute Online Softmax and NO Output Rescaling.
    onlineSoftmaxAndRescale<true, SoftType>(rowMax, rowSum, tSrS, tOrO, scale, tSgSCounting, gmemLayoutS.shape());
  } else { // Compute Online Softmax and Output Rescaling.
    onlineSoftmaxAndRescale<false, SoftType>(rowMax, rowSum, tSrS, tOrO, scale, tSgSCounting, gmemLayoutS.shape());
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
