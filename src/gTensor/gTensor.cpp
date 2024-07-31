#include "gTensor.h"
#include "gTensorIterator.h"
#include "common.h"
#include <cmath>

namespace gblas {

GTensor::GTensor(TSizeArr sizes, TStrideArr strides, unsigned int rank, DType dtype, Layout layout)
       : m_sizes(sizes), m_strides(strides), m_rank(rank), m_dtype(dtype), m_layout(layout)
{}

void GTensor::initData(void* data)
{
    m_buffer = {getMemorySizeInBytes(), (byte*)data};
}

uint64_t GTensor::getTotalSizeInElements() const
{
    uint64_t totalSize = 1;
    for (unsigned i = 0; i < getRank(); i++)
    {
        totalSize *= getSize(i);
    }
    return totalSize;
}

uint64_t GTensor::getMemorySizeInBytes() const
{
    if (m_sizes.empty()) return 0;
    // calculate the max offset available
    uint64_t max_offset = 0;
    for (unsigned i = 0; i < m_rank; ++i)
    {
        max_offset += (getSize(i) - 1) * std::abs(getStride(i));
    }
    // add the element in the max offset and move from elements to bytes
    return (max_offset + 1) * getSingleElementSizeInBytes(getDType());
}

gTensorIterator GTensor::getIterator()
{
    return gTensorIterator(*this);
}

byte *GTensor::operator[](int offset)
{
    uint64_t offsetInBytes = offset * getSingleElementSizeInBytes(getDType());
    return m_buffer[offsetInBytes];
}

byte *GTensor::operator[](Coordinates coords)
{
    uint64_t offsetInElements = 0;
    if (coords.size() != getRank()) throw std::invalid_argument("Coordinates should have the same rank as the tensor");
    for(unsigned idx = 0; idx < coords.size(); ++idx)
    {
        if (coords[idx] >= getSize(idx)) throw std::out_of_range("coordinate is out of bound");
        offsetInElements += coords[idx] * getStride(idx);
    }
    return m_buffer[offsetInElements * getSingleElementSizeInBytes(getDType())];
}




} // gblas