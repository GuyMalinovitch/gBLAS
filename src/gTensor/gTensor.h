#ifndef GBLAS_GTENSOR_H
#define GBLAS_GTENSOR_H

#include <array>
#include "common.h"
#include "DataBuffer.h"

namespace gblas {

class GTensor
{
public:
    GTensor() = default;
    GTensor(TSizeArr sizes, TStrideArr strides, unsigned rank, DType dtype) : m_sizes(sizes), m_strides(strides), m_rank(rank), m_dtype(dtype) {}
    ~GTensor() = default;
    void initData(void* data);
    /// get tensor traits
    DType getDType() const {return m_dtype;}
    const TSizeArr& getAllSizesInElements() const {return m_sizes;}
    uint64_t getSize(unsigned idx) const {return m_sizes[idx];}
    uint64_t getTotalSizeInElements() const;
    uint64_t getTotalSizeInBytes() const;
    uint64_t getMemorySizeInBytes() const;
    const TStrideArr& getAllStridesInElements() const {return m_strides;}
    int64_t getStride(unsigned idx) const {return m_strides[idx];}
    unsigned getRank() const {return m_rank;}

    /// get data
    const DataBuffer* getDataBuffer() const {return &m_buffer;}
    DataBuffer* getDataBuffer() {return &m_buffer;}
private:
    TSizeArr m_sizes = {0};
    TStrideArr m_strides = {0};
    unsigned m_rank = 0;
    DType m_dtype = DType::dtypeNR;
    DataBuffer m_buffer;
};

} // gblas

#endif //GBLAS_GTESNOR_H
