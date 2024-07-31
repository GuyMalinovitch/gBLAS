#ifndef GBLAS_GTENSOR_H
#define GBLAS_GTENSOR_H

#include <array>
#include <stdexcept>
#include "DataBuffer.h"
#include "common.h"

namespace gblas {
class gTensorIterator;

class GTensor
{
public:
    GTensor() = default;
    GTensor(TSizeArr sizes, TStrideArr strides, unsigned rank, DType dtype, Layout layout = Layout::RowMajor);
    ~GTensor() = default;
    GTensor(const GTensor& other) = default;
    bool operator==(const GTensor& other) const = default;
    bool operator!=(const GTensor& other) const = default;
    byte* operator[](int offset);
    byte* operator[](Coordinates coords);
    void initData(void* data);
    /// get tensor traits
    DType getDType() const {return m_dtype;}
    const TSizeArr& getAllSizesInElements() const {return m_sizes;}
    uint64_t getSize(unsigned idx) const {return m_sizes[idx];}
    uint64_t getTotalSizeInElements() const;
    uint64_t getMemorySizeInBytes() const;
    const TStrideArr& getAllStridesInElements() const {return m_strides;}
    int64_t getStride(unsigned idx) const {return m_strides[idx];}
    unsigned getRank() const {return m_rank;}
    Layout getLayout() const {return m_layout;}
    /// get data
    const DataBuffer* getDataBuffer() const {return &m_buffer;}
    DataBuffer* getDataBuffer() {return &m_buffer;}
    gTensorIterator getIterator();
private:
    TSizeArr m_sizes = {1, 1, 1, 1, 1};
    TStrideArr m_strides = {1, 1, 1, 1, 1};
    unsigned m_rank = 1;
    DType m_dtype = DType::dtypeNR;
    Layout m_layout = Layout::LayoutNR;
    DataBuffer m_buffer;
};

} // gblas

#endif //GBLAS_GTESNOR_H
