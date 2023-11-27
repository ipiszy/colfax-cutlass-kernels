/usr/local/cuda-12.2/bin/nvcc -ccbin=/usr/bin/clang++ -use_fast_math -forward-unknown-to-host-compiler -DCUTLASS_ENABLE_CUBLAS=1 -DFMHA -DQBLKSIZE=$1 -DKBLKSIZE=$2 -DHEAD=$3   -I../../lib/ -I ../../include  -I/home/bikshang/repos/cutlass-3.3/cutlass/include -I/home/bikshang/repos/cutlass-3.3/cutlass/examples/common -I"/usr/local/cuda-12.2/include" -I/include -I/examples -I/home/bikshang/repos/cutlass-3.3/cutlass/tools/util/include -O3 -DNDEBUG --generate-code=arch=compute_90a,code=[sm_90a]  -Xcompiler=-fPIE -DCUTLASS_ENABLE_TENSOR_CORE_MMA=1  --expt-extended-lambda --expt-relaxed-constexpr -DCUTLASS_TEST_LEVEL=0 -DCUTLASS_TEST_ENABLE_CACHED_RESULTS=1 -DCUTLASS_CONV_UNIT_TEST_RIGOROUS_SIZE_ENABLED=1 -DCUTLASS_DEBUG_TRACE_LEVEL=0 -Xcompiler=-Wconversion -Xcompiler=-fno-strict-aliasing -Xnvlink=--verbose -Xptxas=--verbose -std=c++17 -MD -MT -MF -x cu  fmha_forward.cu -Wl,-rpath,'/usr/local/cuda-12.2/lib64' -Wl,-rpath,'/usr/local/cuda-12.2/lib' -lcuda  -lcudadevrt -lcudart_static -lcublas -lrt -lpthread -ldl -o fmha_forward_opt_fmha
