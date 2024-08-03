#ifndef GBLAS_FLOAT8_H
#define GBLAS_FLOAT8_H

#include "conversions.h"
#include <cstdint>

namespace gblas {

class fp8_152
{
public:
    fp8_152() = default;

    fp8_152(float val, RoundingMode rounding = RoundingMode::NearestEven)
    {
        m_value = Conversions::fp32_to_fp8_152(val, rounding);
    }
    explicit fp8_152(uint8_t bitarray) { m_value = bitarray; }
    ~fp8_152() = default;
    fp8_152(const fp8_152 &other) { m_value = other.m_value; }
    fp8_152& operator=(const fp8_152 &other) = default;
    uint8_t& value() { return m_value; }
    const uint8_t& value() const { return m_value; }
    float toFloat() const { return Conversions::fp8_152_to_fp32(m_value); }

    // relational operators
    bool operator<(const float& rhs) const { return toFloat() < rhs; }
    bool operator>(const float& rhs) const { return toFloat() > rhs; }
    bool operator<(const fp8_152& rhs) const { return value() < rhs.value(); }
    bool operator>(const fp8_152& rhs) const { return value() > rhs.value(); }
    bool operator==(const fp8_152& rhs) const { return value() == rhs.value(); }
    bool operator!=(const fp8_152& rhs) const { return value() != rhs.value(); }
    bool operator==(const float& rhs) const { return toFloat() == rhs; }
    bool operator!=(const float& rhs) const { return toFloat() != rhs; }

    //casting operators
    explicit operator float() const { return toFloat(); }
    explicit operator double() const { return toFloat(); }
    explicit operator uint8_t() const { return m_value; }

    // identify special values
    bool isZero() const { return (m_value == 0x0 || m_value == 0x80); }
    static bool isZero(const fp8_152& val) { return val.isZero(); }
    bool isInf() const { return (m_value == 0x7C || m_value == 0xFC); }
    static bool isInf(const fp8_152& val) { return val.isInf(); }
    bool isNan() const
    {
        uint8_t exponent = (m_value >> 2) & 0x1F;
        uint8_t mantissa = m_value & 0x3;
        return (exponent == 0x1F) && (mantissa != 0);
    }
    static bool isNan(const fp8_152& val) { return val.isNan(); }
    static fp8_152 max() { return fp8_152((uint8_t)0x7B); }
    static fp8_152 min() { return fp8_152((uint8_t)0x04); }
    static fp8_152 lowest() { return fp8_152((uint8_t)0xFB); }
private:
    uint8_t m_value;
};

class fp8_143
{
public:
    fp8_143() = default;

    fp8_143(float val, RoundingMode rounding = RoundingMode::NearestEven)
    {
        m_value = Conversions::fp32_to_fp8_143(val, rounding);
    }
    explicit fp8_143(uint8_t bitarray) { m_value = bitarray; }
    ~fp8_143() = default;
    fp8_143(const fp8_143 &other) { m_value = other.m_value; }
    fp8_143& operator=(const fp8_143 &other) = default;
    uint8_t& value() { return m_value; }
    const uint8_t& value() const { return m_value; }
    float toFloat() const { return Conversions::fp8_143_to_fp32(m_value); }

    // relational operators
    bool operator<(const float& rhs) const { return toFloat() < rhs; }
    bool operator>(const float& rhs) const { return toFloat() > rhs; }
    bool operator<(const fp8_143& rhs) const { return value() < rhs.value(); }
    bool operator>(const fp8_143& rhs) const { return value() > rhs.value(); }
    bool operator==(const fp8_143& rhs) const { return value() == rhs.value(); }
    bool operator!=(const fp8_143& rhs) const { return value() != rhs.value(); }
    bool operator==(const float& rhs) const { return toFloat() == rhs; }
    bool operator!=(const float& rhs) const { return toFloat() != rhs; }

    //casting operators
    explicit operator float() const { return toFloat(); }
    explicit operator double() const { return toFloat(); }
    explicit operator uint8_t() const { return m_value; }

    // identify special values
    bool isZero() const { return (m_value == 0x0 || m_value == 0x80); }
    static bool isZero(const fp8_143& val) { return val.isZero(); }
    bool isInf() const { return (m_value == 0x78 || m_value == 0xF8); }
    static bool isInf(const fp8_143& val) { return val.isInf(); }
    bool isNan() const
    {
        uint8_t exponent = (m_value >> 3) & 0xF;
        uint8_t mantissa = m_value & 0x7;
        return (exponent == 0xF) && (mantissa != 0);
    }
    static bool isNan(const fp8_143& val) { return val.isNan(); }
    static fp8_143 max() { return fp8_143((uint8_t)0x77); }
    static fp8_143 min() { return fp8_143((uint8_t)0x08); }
    static fp8_143 lowest() { return fp8_143((uint8_t)0xF7); }
private:
    uint8_t m_value;
};

static_assert(sizeof(fp8_152) == sizeof(uint8_t), "size of fp8_152 must be 8 bits for reinterpret_cast to work");
static_assert(sizeof(fp8_143) == sizeof(uint8_t), "size of fp8_143 must be 8 bits for reinterpret_cast to work");

} // namespace gblas
#endif //GBLAS_FLOAT8_H
