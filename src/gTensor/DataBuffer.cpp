#include "DataBuffer.h"
#include <iostream>
#include <cassert>

namespace gblas {

DataBuffer::DataBuffer(uint64_t size)
{
    m_buffer = new byte[size];
    m_size = size;
    shouldFreeOnDtor = true;
}

DataBuffer::DataBuffer(uint64_t size, byte* data)
{
    m_buffer = data;
    m_size = size;
    shouldFreeOnDtor = true;
}

DataBuffer::~DataBuffer()
{
    if (shouldFreeOnDtor && m_buffer)
    {
        delete[](m_buffer);
    }
}

DataBuffer::DataBuffer(const DataBuffer &other) : m_size(other.m_size)
{
    if (other.m_buffer)
    {
        m_buffer = new byte[m_size];
        assert(m_buffer);
        std::memcpy(m_buffer, other.m_buffer, m_size);
        shouldFreeOnDtor = true;
    }
}

DataBuffer &DataBuffer::operator=(const DataBuffer &other)
{
    if (this != &other)
    {
        if (shouldFreeOnDtor && m_buffer)
        {
            delete[](m_buffer);
        }
        m_buffer = nullptr;
        m_size = other.m_size;
        if (other.m_buffer)
        {
            m_buffer = new byte[m_size];
            assert(m_buffer);
            std::memcpy(m_buffer, other.m_buffer, m_size);
            shouldFreeOnDtor = true;
        }
    }
    return *this;
}

DataBuffer::DataBuffer(DataBuffer &&other) noexcept: m_buffer(other.m_buffer), m_size(other.m_size), shouldFreeOnDtor(other.shouldFreeOnDtor)
{
    other.m_buffer = nullptr;
    other.m_size = 0;
}

DataBuffer &DataBuffer::operator=(DataBuffer &&other) noexcept
{
    if (this != &other)
    {
        if (shouldFreeOnDtor && m_buffer)
        {
            free(m_buffer);
        }

        m_buffer = other.m_buffer;
        m_size = other.m_size;
        shouldFreeOnDtor = other.shouldFreeOnDtor;

        other.m_buffer = nullptr;
        other.m_size = 0;
    }
    return *this;
}

} // gblas