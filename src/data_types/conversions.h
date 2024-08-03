
#ifndef GBLAS_CONVERSIONS_H
#define GBLAS_CONVERSIONS_H

#include <stdint.h>
#include <cmath>
#include <cstring>

namespace gblas {
template<typename T, typename U>
concept reinterpretRestriction = requires {sizeof(T) == sizeof(U) && std::is_const<T>() == std::is_const<U>();};

enum class RoundingMode
{
    NearestEven,
    RoundUp,
    RoundDown,
    RoundAwayFromZero,
    RoundTowardsZero
};

class Conversions
{
public:
    // reinterpret a pointer of Source type and copy it to value of Target type
    template<typename Dest, typename Source> requires reinterpretRestriction<Dest, Source>
    static Dest reinterpret_ptr(Source* val)
    {
        return *reinterpret_cast<Dest*>(val);
    }
    static uint16_t fp32_to_bf16(const float& val, RoundingMode rounding)
    {
        uint16_t result;
        auto floatInBits = reinterpret_ptr<const uint32_t>(&val);
        // need to extract the first 16 bits , and round
        result = static_cast<uint16_t>(floatInBits >> 16);
        uint32_t lowerBits = floatInBits & 0xFFFF;
        bool isPositive = (floatInBits & 0x80000000) == 0;
        switch (rounding)
        {
            case RoundingMode::NearestEven:
                if (lowerBits > 0x8000) result++;
                // tiebreaker - check if value is odd\even
                if (lowerBits == 0x8000 && (result & 1)) result++;
            case RoundingMode::RoundUp:
                if (lowerBits != 0 && isPositive)
                {
                    result++;
                }
                break;
            case RoundingMode::RoundDown:
                // round only for negative numbers, for positive truncation is enough.
                if (!isPositive && lowerBits != 0)
                {
                    result++;
                }
                break;
            case RoundingMode::RoundAwayFromZero:
                if (lowerBits != 0)
                {
                    if (isPositive) result++;
                    // check we don't increment -0 case.
                    if (result != 0x8000) result++;
                }
                break;
            case RoundingMode::RoundTowardsZero:
                // truncation is enough
                break;
        }
        return result;
    }
    static float bf16_to_fp32(const uint16_t& valAsBits)
    {
        uint32_t floatAsBits = (uint32_t)valAsBits << 16;
        return reinterpret_ptr<const float, const uint32_t>(&floatAsBits);
    }

    static uint16_t fp32_to_fp16(const float& val, RoundingMode rounding)
    {
        uint16_t result;
        auto floatInBits = reinterpret_ptr<const uint32_t>(&val);

        // Extract components of fp32
        uint32_t sign = floatInBits >> 31;
        uint32_t exponent = (floatInBits >> 23) & 0xFF;
        uint32_t mantissa = floatInBits & 0x7FFFFF;

        // Handle special cases
        if (std::isinf(val))
        {
            return sign ? 0xFC00 : 0x7C00;
        }
        if (std::isnan(val))
        {
            return sign ? 0xFE00 : 0x7E00;
        }
        if (val == 0.0f)
        {
            result = (sign ? 0x8000 : 0x0000);
            return result;
        }

        // Adjust for fp16 exponent bias
        int32_t adjustedExponent = static_cast<int32_t>(exponent) - 127 + 15;

        if (adjustedExponent > 30) // Overflow to infinity
        {
            result = (sign ? 0xFC00 : 0x7C00);
            return result;
        }
        if (adjustedExponent < -14) // Underflow to subnormal or zero
        {
            result = sign << 15;
            return result;
        }

        // Prepare the number
        uint32_t roundBit = 0;
        uint32_t stickyBit = 0;
        if (adjustedExponent >= 1) {  // Normal number
            roundBit = (mantissa >> 12) & 1;
            stickyBit = (mantissa & 0xFFF) != 0;
            mantissa >>= 13;
        } else {  // Subnormal number
            mantissa |= 0x800000;  // Add implicit leading 1
            int32_t shift = 14 - adjustedExponent;
            roundBit = (mantissa >> shift) & 1;
            stickyBit = (mantissa & ((1 << shift) - 1)) != 0;
            mantissa >>= (shift + 1);
            adjustedExponent = 0;
        }

        uint16_t fp16Mantissa = mantissa & 0x3FF;
        uint16_t fp16Exponent = adjustedExponent & 0x1F;

        // Apply rounding
        bool shouldRoundUp = false;
        switch (rounding) {
            case RoundingMode::NearestEven:
                shouldRoundUp = roundBit && (stickyBit || (fp16Mantissa & 1));
                break;
            case RoundingMode::RoundUp:
                shouldRoundUp = !sign && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundDown:
                shouldRoundUp = sign && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundTowardsZero:
                shouldRoundUp = false;
                break;
            case RoundingMode::RoundAwayFromZero:
                shouldRoundUp = roundBit || stickyBit;
                break;
        }

        if (shouldRoundUp) {
            fp16Mantissa++;
            if (fp16Mantissa == 0x400) {  // Mantissa overflow
                fp16Mantissa = 0;
                fp16Exponent++;
                if (fp16Exponent == 0x1F) {  // Exponent overflow
                    fp16Exponent = 0x1E;
                    fp16Mantissa = 0x3FF;
                }
            }
        }
        // Compose the fp16
        result = (sign << 15) | (fp16Exponent << 10) | fp16Mantissa;
        return result;
    }
    static float fp16_to_fp32(const uint16_t& valAsBits)
    {
        // Extract components of fp16
        uint32_t sign = valAsBits >> 15;
        uint32_t exponent = (valAsBits >> 10) & 0x1F;
        uint32_t mantissa = valAsBits & 0x3FF;

        uint32_t floatInBits;

        if (exponent == 0)
        {  // Zero or subnormal number
            if (mantissa == 0)
            {
                // Zero
                return sign ? -0.0f : 0.0f;
            }
            else
            {
                // Subnormal number -- renormalize
                while (!(mantissa & 0x400))
                {
                    mantissa <<= 1;
                    exponent -= 1;
                }
                exponent += 1;
                mantissa &= ~0x400;
            }
        }
        else if (exponent == 31)
        {
            if (mantissa == 0)
            {
                // Infinity
                return sign ? -INFINITY : INFINITY;
            }
            else
            {
                return NAN;
            }
        }
        // Convert to fp32 and adjust exponent
        floatInBits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 13);

        return reinterpret_ptr<float>(&floatInBits);
    }
    static uint32_t fp32_to_tf32(const float& val, RoundingMode rounding)
    {
        auto floatInBits = reinterpret_ptr<const uint32_t>(&val);

        // Extract components of fp32
        uint32_t sign = floatInBits >> 31;
        uint32_t exponent = (floatInBits >> 23) & 0xFF;
        uint32_t mantissa = floatInBits & 0x7FFFFF;

        uint32_t result;

        // Handle special cases: ±0, ±inf, NaN
        if (exponent == 0 || exponent == 0x7F800000)
        {
            result = floatInBits; // These cases are identical in tf32
            return result;
        }

        // Round the mantissa
        uint32_t roundBit = (mantissa >> 13) & 1;
        uint32_t stickyBit = (mantissa & 0x1FFF) != 0;

        bool shouldRoundUp = false;
        switch (rounding)
        {
            case RoundingMode::NearestEven:
                shouldRoundUp = roundBit && (stickyBit || ((mantissa >> 14) & 1));
                break;
            case RoundingMode::RoundUp:
                shouldRoundUp = (sign == 0) && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundDown:
                shouldRoundUp = (sign != 0) && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundTowardsZero:
                shouldRoundUp = false;
                break;
            case RoundingMode::RoundAwayFromZero:
                shouldRoundUp = roundBit || stickyBit;
                break;
        }

        mantissa >>= 13; // keep top 10 bits of mantissa for tf32
        if (shouldRoundUp)
        {
            mantissa++;
            if (mantissa == 0x400)
            { // Mantissa overflow
                mantissa = 0;
                exponent += 0x00800000;
            }
        }

        // Compose the tf32
        result = sign | exponent | mantissa;
        return result;
    }
    static float tf32_to_fp32(const uint32_t& valAsBits)
    {
        return reinterpret_ptr<const float>(&valAsBits);
    }

    static uint8_t fp32_to_fp8_152(const float& val, RoundingMode rounding)
    {
        auto floatInBits = reinterpret_ptr<const uint32_t>(&val);

        // Extract components of fp32
        uint32_t sign = floatInBits >> 31;
        uint32_t exponent = (floatInBits >> 23) & 0xFF;
        uint32_t mantissa = floatInBits & 0x7FFFFF;

        uint8_t result;

        // Handle special cases: ±0, ±inf, NaN
        if (std::isinf(val))
        {
            return sign ? 0xFC :0x7C;
        }
        if (std::isnan(val))
        {
            return sign ? 0xFE : 0x7E;
        }
        // Adjust exponent (fp32 bias: 127, fp8 bias: 15)
        int16_t adjustedExponent = exponent - 127 + 15;  // fp8_152 has bias 15

        // Handle underflow and overflow
        if (adjustedExponent < 1)
        {
            result = sign << 7;  // Underflow to zero
            return result;
        }
        if (adjustedExponent > 30)
        {
            result = (sign << 7) | 0x7F;  // Overflow to ±inf
            return result;
        }

        // Prepare mantissa for rounding (keep top 2 bits for fp8)
        uint32_t roundBit = (mantissa >> 20) & 1;
        uint32_t stickyBit = (mantissa & 0x1FFFFF) != 0;
        mantissa >>= 21;  // Keep top 2 bits for fp8 mantissa

        // Apply rounding
        bool shouldRoundUp = false;
        switch (rounding)
        {
            case RoundingMode::NearestEven:
                shouldRoundUp = roundBit && (stickyBit || (mantissa & 1));
                break;
            case RoundingMode::RoundUp:
                shouldRoundUp = !sign && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundDown:
                shouldRoundUp = sign && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundTowardsZero:
                shouldRoundUp = false;
                break;
            case RoundingMode::RoundAwayFromZero:
                shouldRoundUp = roundBit || stickyBit;
                break;
        }

        if (shouldRoundUp) {
            mantissa++;
            if (mantissa == 4) {  // Mantissa overflow
                mantissa = 0;
                adjustedExponent++;
                if (adjustedExponent > 30) {
                    result = (sign << 7) | 0x7F;  // Overflow to ±inf
                    return result;
                }
            }
        }

        // Compose the fp8 (1 sign bit, 5 exponent bits, 2 mantissa bits)
        result = (sign << 7) | (adjustedExponent << 2) | mantissa;
        return result;
    }
    static uint8_t fp32_to_fp8_143(const float& val, RoundingMode rounding)
    {
        auto floatInBits = reinterpret_ptr<const uint32_t>(&val);

        // Extract components of fp32
        uint32_t sign = floatInBits >> 31;
        uint32_t exponent = (floatInBits >> 23) & 0xFF;
        uint32_t mantissa = floatInBits & 0x7FFFFF;

        uint8_t result;

        // Handle special cases: ±0, ±inf, NaN
        if (std::isinf(val))
        {
            return sign ? 0xF8 :0x78;
        }
        if (std::isnan(val))
        {
            return sign ? 0xFC : 0x7C;
        }

        // Adjust exponent (fp32 bias: 127, fp8 bias: 7)
        int16_t adjustedExponent = exponent - 127 + 7;

        // Handle underflow and overflow
        if (adjustedExponent < 1)
        {
            result = sign << 7;  // Underflow to zero
            return result;
        }
        if (adjustedExponent > 14)
        {
            result = (sign << 7) | 0x7F;  // Overflow to ±inf
            return result;
        }

        // Prepare mantissa for rounding (keep top 3 bits for fp8)
        uint32_t roundBit = (mantissa >> 19) & 1;
        uint32_t stickyBit = (mantissa & 0xFFFFF) != 0;
        // Keep 3 leading bits
        mantissa >>= 20;

        // Apply rounding
        bool shouldRoundUp = false;
        switch (rounding)
        {
            case RoundingMode::NearestEven:
                shouldRoundUp = roundBit && (stickyBit || (mantissa & 1));
                break;
            case RoundingMode::RoundUp:
                shouldRoundUp = !sign && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundDown:
                shouldRoundUp = sign && (roundBit || stickyBit);
                break;
            case RoundingMode::RoundTowardsZero:
                shouldRoundUp = false;
                break;
            case RoundingMode::RoundAwayFromZero:
                shouldRoundUp = roundBit || stickyBit;
                break;
        }

        if (shouldRoundUp)
        {
            mantissa++;
            if (mantissa == 8) {  // Mantissa overflow
                mantissa = 0;
                adjustedExponent++;
                if (adjustedExponent > 14) {
                    result = (sign << 7) | 0x7F;  // Overflow to ±inf
                    return result;
                }
            }
        }

        // Compose the fp8 (1 sign bit, 4 exponent bits, 3 mantissa bits)
        result = (sign << 7) | (adjustedExponent << 3) | mantissa;
        return result;
    }
    static float fp8_152_to_fp32(const uint8_t& valAsBits)
    {
        uint32_t sign = valAsBits >> 7;
        uint32_t exponent = (valAsBits >> 2) & 0x1F;
        uint32_t mantissa = valAsBits & 0x3;
        uint32_t floatInBits;
        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                // Zero
                return sign ? -0.0f : 0.0f;
            }
            else
            {
                // Subnormal number
                exponent = 1;
                while (!(mantissa & 0x4)) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x3;
                floatInBits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 21);
            }
        }
        else if (exponent == 31)
        {
            if (mantissa == 0)
            {
                // Infinity
                return sign ? -INFINITY : INFINITY;
            }
            else
            {
                // NaN
                return NAN;
            }
        }
        else
        {
            // Normal number
            floatInBits = (sign << 31) | ((exponent + 112) << 23) | (mantissa << 21);
        }
        return reinterpret_ptr<float>(&floatInBits);
    }
    static float fp8_143_to_fp32(const uint8_t& valAsBits)
    {
        uint32_t sign = valAsBits >> 7;
        uint32_t exponent = (valAsBits >> 3) & 0xF;
        uint32_t mantissa = valAsBits & 0x7;
        uint32_t floatInBits;

        if (exponent == 0)
        {
            if (mantissa == 0)
            {
                // Zero
                return sign ? -0.0f : 0.0f;
            }
            else
            {
                // Subnormal number
                exponent = 1;
                while (!(mantissa & 0x8)) {
                    mantissa <<= 1;
                    exponent--;
                }
                mantissa &= 0x7;
                floatInBits = (sign << 31) | ((exponent + 120) << 23) | (mantissa << 20);
            }
        }
        else if (exponent == 0xF)
        {
            if (mantissa == 0)
            {
                // Infinity
                return sign ? -INFINITY : INFINITY;
            } else {
                // NaN
                return NAN;
            }
        }
        else
        {
            // Normal number
            floatInBits = (sign << 31) | ((exponent + 120) << 23) | (mantissa << 20);
        }
        return reinterpret_ptr<float>(&floatInBits);
    }

};


} // namespace gblas
#endif //GBLAS_CONVERSIONS_H
