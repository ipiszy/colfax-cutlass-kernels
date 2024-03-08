#pragma once

#include "cutlass/util/distribution.h"
#include "cutlass/util/reference/device/tensor_fill.h"

namespace cfk {
template <typename TIN, typename TOUT = TIN>
void initialize_rand(TIN *ptr, size_t capacity,
                     cutlass::Distribution::Kind dist_kind, uint32_t seed) {

  if (dist_kind == cutlass::Distribution::Uniform) {

    TIN scope_max, scope_min;
    int bits_input = cutlass::sizeof_bits<TIN>::value;
    int bits_output = cutlass::sizeof_bits<TOUT>::value;

    if (bits_input == 1) {
      scope_max = TIN(2);
      scope_min = TIN(0);
    } else if (bits_input <= 8) {
      scope_max = TIN(2);
      scope_min = TIN(-2);
    } else if (bits_output == 16) {
      scope_max = TIN(8);
      scope_min = TIN(-8);
    } else {
      scope_max = TIN(8);
      scope_min = TIN(-8);
    }

    cutlass::reference::device::BlockFillRandomUniform(ptr, capacity, seed,
                                                       scope_max, scope_min, 0);
  } else if (dist_kind == cutlass::Distribution::Gaussian) {

    cutlass::reference::device::BlockFillRandomGaussian(ptr, capacity, seed,
                                                        TIN(), TIN(1.0f));
  } else if (dist_kind == cutlass::Distribution::Sequential) {

    // Fill with increasing elements
    cutlass::reference::device::BlockFillSequential(ptr, capacity, TIN(1),
                                                    TIN());
  }
}

template <typename Element>
void initialize_const(Element *ptr, size_t capacity, const Element &value) {

  // Fill with all 1s
  cutlass::reference::device::BlockFillSequential(ptr, capacity, Element(),
                                                  value);
}

template <typename Element>
bool verify_tensor(thrust::host_vector<Element> vector_Input,
                   thrust::host_vector<Element> vector_Input_Ref,
                   int batchSize, int m, int dim,
                   bool printValues = false, bool printDiffs = false,
                   float errCountExpected = 0, int64_t verify_length = -1,
                   bool useVarSeqLength = false,
                   thrust::host_vector<uint64_t> hostSeqOffsets = thrust::host_vector<uint64_t>()
) {
  int64_t size = (vector_Input.size() < vector_Input_Ref.size())
                     ? vector_Input.size()
                     : vector_Input_Ref.size();
  size = (verify_length == -1) ? size : verify_length;

  // 0.005 for absolute error
  float abs_tol = 5e-3f;
  // 10% for relative error
  float rel_tol = 1e-1f;
  int errCount = 0;
  for (int i = 0; i < batchSize; ++i) {
    int numElements =
        (useVarSeqLength ? (hostSeqOffsets[i + 1] - hostSeqOffsets[i]) : m) * dim;
    uint64_t offset = (useVarSeqLength ? hostSeqOffsets[i] : (i * m)) * dim;
    uint64_t paddedOffset = i * m * dim;
    for (int j = 0; j < numElements; ++j) {
      uint64_t idx = offset + j;
      uint64_t refIdx = paddedOffset + j;
      if (printValues)
        std::cout << vector_Input[idx] << " " << vector_Input_Ref[refIdx] << std::endl;
      float diff = (float)(vector_Input[idx] - vector_Input_Ref[refIdx]);
      if (
        (float(vector_Input[idx]) == std::numeric_limits<float>::infinity() &&
          float(vector_Input_Ref[refIdx]) == std::numeric_limits<float>::infinity()) ||
        (float(vector_Input[idx]) == -std::numeric_limits<float>::infinity() &&
          float(vector_Input_Ref[refIdx]) == -std::numeric_limits<float>::infinity())
      ) {
        diff = 0;
      }
      float abs_diff = fabs(diff);
      float abs_ref = fabs((float)vector_Input_Ref[refIdx] + 1e-5f);
      float relative_diff = abs_diff / abs_ref;
      if ((isnan(vector_Input_Ref[idx]) || isnan(abs_diff) || isinf(abs_diff)) ||
          (abs_diff > abs_tol && relative_diff > rel_tol)) {
        if (printDiffs)
          printf("[%d/%d] diff = %f, rel_diff = %f, {computed=%f, ref=%f}.\n",
                 int(idx), int(size), abs_diff, relative_diff,
                 (float)(vector_Input[idx]), (float)(vector_Input_Ref[refIdx]));
        errCount++;
        // return false;
      }
    }
  }
  auto errCountComputed = float(errCount) / float(size) * 100;
  printf("Error (percentage) : %f\n", errCountComputed);
  return errCountComputed <= errCountExpected ? true : false;
}

} // namespace cfk
