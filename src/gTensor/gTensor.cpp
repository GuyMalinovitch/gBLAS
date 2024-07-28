#include "gTensor.h"
#include <cmath>

namespace gblas {

void GTensor::initData(void* data)
{
    m_buffer = {getTotalSizeInBytes(), (byte*)data};
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

uint64_t GTensor::getTotalSizeInBytes() const
{
    return getTotalSizeInElements() * getSingleElementSizeInBytes(getDType());
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


} // gblas