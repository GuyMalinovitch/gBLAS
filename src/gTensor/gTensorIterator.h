#ifndef GBLAS_GTENSORITERATOR_H
#define GBLAS_GTENSORITERATOR_H

#include <stdexcept>
#include "gTensor.h"

namespace gblas
{

class gTensorIterator
{
public:
    explicit gTensorIterator(gTensor& tensor) : m_tensor(tensor),
                                                m_totalNumOfElements(tensor.getTotalSizeInElements()) {}
    ~gTensorIterator() = default;
    gTensorIterator(const gTensorIterator& other) : m_tensor(other.m_tensor)
    {
        m_totalNumOfElements = other.m_totalNumOfElements;
        m_currentIndex = other.m_currentIndex;
    }

    gTensorIterator& operator++() noexcept
    {
        if (m_currentIndex != m_totalNumOfElements+1)
        {
            m_currentIndex++;
        }
        return *this;
    }

    const gTensorIterator operator++(int) noexcept
    {
        auto tempIt = *this;
        ++*this;
        return tempIt;
    }

    bool operator==(const gTensorIterator& other) const
    {
        return (m_tensor == other.m_tensor ||
                m_currentIndex == other.m_currentIndex);
    }
    bool operator!=(const gTensorIterator& other) const = default;

    gTensorIterator& begin()
    {
        m_currentIndex = 0;
        return *this;
    }
     gTensorIterator& end()
     {
        m_currentIndex = m_totalNumOfElements + 1;
        return *this;
     }
     byte* operator*() const noexcept
     {
        return m_tensor[m_currentIndex];
     }
     Coordinates getCurrentCoordinates()
     {
        Coordinates coords;
        uint64_t offset = m_currentIndex;
        auto& strides = m_tensor.getAllStridesInElements();
        std::fill(coords.begin(), coords.end(), 1);
        for(unsigned i = m_tensor.getRank()-1; 0 <= i; --i)
        {
            coords[i] = offset / strides[i];
            offset %= strides[i];
            if (coords[i] >= m_tensor.getSize(i)) throw std::out_of_range("Offset is out of bound");
        }
        if (offset != 0)
        {
            throw std::out_of_range("Offset is out of bound");
        }
     }
private:
    gTensor& m_tensor;
    uint64_t m_currentIndex = 0;
    uint64_t m_totalNumOfElements = 0;
};

} // namespace gblas
#endif //GBLAS_GTENSORITERATOR_H
