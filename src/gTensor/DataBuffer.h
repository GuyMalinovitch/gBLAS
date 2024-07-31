#ifndef GBLAS_DATABUFFER_H
#define GBLAS_DATABUFFER_H
#include <cstdint>
#include <cstring>

namespace gblas {
using byte = uint8_t;

class DataBuffer {
public:
    DataBuffer() = default;
    explicit DataBuffer(uint64_t sizeInBytes);
    DataBuffer(uint64_t sizeInBytes, byte* data);
    ~DataBuffer();
    DataBuffer(const DataBuffer& other);
    DataBuffer& operator=(const DataBuffer& other);
    DataBuffer(DataBuffer&& other) noexcept;
    DataBuffer& operator=(DataBuffer&& other) noexcept;
    bool operator==(const DataBuffer& other) const
    {
        return (m_size == other.m_size) && (std::memcmp(m_buffer, other.m_buffer, m_size) == 0);
    }
    bool operator!=(const DataBuffer& other) const
    {
        return !operator==(other);
    }
    byte* operator[](unsigned i) {return &m_buffer[i];}
    const byte* operator[](unsigned i) const {return &m_buffer[i];}
private:
    bool shouldFreeOnDtor = false;
    byte* m_buffer = nullptr;
    unsigned m_size = 0;
};




} // gblas

#endif //GBLAS_DATABUFFER_H
