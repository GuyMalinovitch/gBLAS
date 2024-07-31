#ifndef GBLAS_AXPY_H
#define GBLAS_AXPY_H

#include <cstring>
#include <cstdint>

namespace gblas {
class GTensor;
enum class gStatus;

class Operations
{
public:
    Operations() = default;
    ~Operations() = default;
    // Level 1 operations //
    // perform a*X+Y operation
    template<typename T>
    gStatus axpy(uint64_t a, const GTensor& x, const GTensor& y, GTensor& out, bool transposeX = false, bool transposeY = false);
};




} //namespace gblas
#endif //GBLAS_AXPY_H