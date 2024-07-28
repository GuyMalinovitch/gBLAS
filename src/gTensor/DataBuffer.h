#ifndef GBLAS_DATABUFFER_H
#define GBLAS_DATABUFFER_H
#include <cstdint>

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

    byte* operator[](int i) {return &m_buffer[i];}
    const byte* operator[](int i) const {return &m_buffer[i];}
    std::string getInfo();
private:
    bool shouldFreeOnDtor = false;
    byte* m_buffer = nullptr;
    unsigned m_size = 0;
};




} // gblas

#endif //GBLAS_DATABUFFER_H
