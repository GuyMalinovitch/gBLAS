#ifndef GBLAS_FLOAT16_H
#define GBLAS_FLOAT16_H

#include "conversions.h"
#include <cstdint>

namespace gblas {


class Float16
{
public:
    Float16() = default;

    Float16(float val, RoundingMode rounding = RoundingMode::NearestEven)
    {
        m_value = Conversions::fp32_to_fp16(val, rounding);
    }
    explicit Float16(uint16_t bitarray) { m_value = bitarray; }
    ~Float16() = default;
    Float16(const Float16 &other) { m_value = other.m_value; }
    Float16 &operator=(const Float16 &other) = default;
    uint16_t& value() { return m_value; }
    const uint16_t& value() const { return m_value; }
    float toFloat() const { return Conversions::fp16_to_fp32(m_value); }

    // relational operators
    bool operator<(const float &rhs) const { return toFloat() < rhs; }
    bool operator>(const float &rhs) const { return toFloat() > rhs; }
    bool operator<(const Float16 &rhs) const { return value() < rhs.value(); }
    bool operator>(const Float16 &rhs) const { return value() > rhs.value(); }
    bool operator==(const Float16 &rhs) const { return value() == rhs.value(); }
    bool operator!=(const Float16 &rhs) const { return value() != rhs.value(); }
    bool operator==(const float &rhs) const { return toFloat() == rhs; }
    bool operator!=(const float &rhs) const { return toFloat() != rhs; }

    //casting operators
    explicit operator float() const { return toFloat(); }
    explicit operator double() const { return toFloat(); }
    explicit operator uint16_t() const { return m_value; }

    // identify special values
    bool isZero() const { return (m_value == 0x0 || m_value == 0x8000); }
    static bool isZero(const Float16 &val) { return val.isZero(); }
    bool isInf() const { return (m_value == 0x7C00 || m_value == 0xFC00); }
    static bool isInf(const Float16 &val) { return val.isInf(); }
    bool isNan() const
    {
        uint8_t exponent = (m_value >> 10) & 0x1F;
        uint16_t mantissa = m_value & 0x3FF;
        return (exponent == 0x1F) && (mantissa != 0);
    }
    static bool isNan(const Float16 &val) { return val.isNan(); }
    static Float16 max() { return Float16((uint16_t)0x7BFF); }
    static Float16 min() { return Float16((uint16_t)0x0400); }
    static Float16 lowest() { return Float16((uint16_t)0xFBFF); }

private:
    uint16_t m_value = 0;
};

static_assert(sizeof(Float16) == sizeof(uint16_t), "size of Float16 must be 16bits for reinterpret_cast to work");
using fp16_t = Float16;

} // namespace gblas

#endif //GBLAS_FLOAT16_H
