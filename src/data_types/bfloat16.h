
#ifndef GBLAS_BFLOAT16_H
#define GBLAS_BFLOAT16_H

#include "conversions.h"
#include <stdint.h>

namespace gblas {
/*
 * @file Implementation for bfloat16 data type
 * 1 Sign bit
 * 8 Exponent bits
 * 7 Mantissa bits
 */

class Bfloat16
{
public:
    Bfloat16() = default;

    Bfloat16(float val, RoundingMode rounding = RoundingMode::NearestEven)
    {
        m_value = Conversions::fp32_to_bf16(val, rounding);
    }
    explicit Bfloat16(uint16_t bitarray) {m_value = bitarray;}
    ~Bfloat16() = default;
    Bfloat16(const Bfloat16& other) {m_value = other.m_value;}
    Bfloat16& operator=(const Bfloat16& other) = default;

    uint16_t& value() {return m_value;}
    const uint16_t& value() const {return m_value;}
    float toFloat() const {return Conversions::bf16_to_fp32(m_value);}
    // relational operators
    bool operator<(const float& rhs) const {return toFloat() < rhs;}
    bool operator>(const float& rhs) const {return toFloat() > rhs;}
    bool operator<(const Bfloat16& rhs) const {return value() < rhs.value();}
    bool operator>(const Bfloat16& rhs) const {return value() > rhs.value();}
    bool operator==(const Bfloat16& rhs) const {return value() == rhs.value();}
    bool operator!=(const Bfloat16& rhs) const {return value() != rhs.value();}
    bool operator==(const float& rhs) const {return toFloat() == rhs;}
    bool operator!=(const float& rhs) const {return toFloat() != rhs;}

    //casting operators
    explicit operator float() const {return toFloat();}
    explicit operator double() const {return toFloat();}
    explicit operator uint16_t() const {return m_value;}

    // identify special values
    bool isZero() const {return (m_value == 0x0 || m_value == 0x8000);}
    static bool isZero(const Bfloat16& val) {return val.isZero();}
    bool isInf() const {return (m_value == 0x7F80 || m_value == 0xFF80);}
    static bool isInf(const Bfloat16& val ) {return val.isInf();}
    bool isNan() const
    {
        uint8_t exponent = (m_value >> 7) & 0xFF;
        uint8_t mantissa = m_value & 0x7F;
        return (exponent == 0xFF) && (mantissa != 0);
    }
    static bool isNan(const Bfloat16& val) { return val.isNan();}
    static Bfloat16 max() {return Bfloat16((uint16_t)0x7F7F);}
    static Bfloat16 min() {return Bfloat16((uint16_t)0x0080);}
    static Bfloat16 lowest() {return Bfloat16((uint16_t)0xFF7F);}
private:
    uint16_t m_value = 0;
};

static_assert(sizeof(Bfloat16) == sizeof(uint16_t), "size of Bfloat16 must be 16bits for reinterpret_cast to work");
using bf16_t = Bfloat16;


} //namespace gblas
#endif //GBLAS_BFLOAT16_H
