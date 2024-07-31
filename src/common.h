#ifndef GBLAS_COMMON_H
#define GBLAS_COMMON_H

#include <cassert>
#include <cstdint>

namespace gblas {

 enum class gStatus
 {
    gBLAS_PASS,
    gBLAS_FAIL
 };

#define MAX_DIM 5
using TSizeArr = std::array<uint64_t, MAX_DIM>;
using TStrideArr = std::array<int64_t, MAX_DIM>;

enum class DType
{
    int8,
    fp8_152,
    fp8_143,
    int16,
    fp16,
    bf16,
    int32,
    fp32,
    tf32,
    int64,
    fp64,
    dtypeNR
};

inline unsigned getSingleElementSizeInBytes(DType dtype)
{
    switch(dtype)
    {

        case DType::int8:
        case DType::fp8_152:
        case DType::fp8_143:
            return 1;
            break;
        case DType::int16:
        case DType::fp16:
        case DType::bf16:
            return 2;
            break;
        case DType::int32:
        case DType::fp32:
        case DType::tf32:
            return 4;
            break;
        case DType::int64:
        case DType::fp64:
            return 8;
            break;
    }
    assert("should not get here !");
    return 0;
}

enum class Layout
{
    RowMajor,
    ColMajor,
    LayoutNR
};

using Coordinates = std::array<unsigned, MAX_DIM>;


} // namespace gblas

#endif //GBLAS_COMMON_H


