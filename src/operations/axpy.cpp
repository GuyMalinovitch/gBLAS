//
// Created by gmalino on 30/07/2024.
//

#include "axpy.h"
#include "gTensor/gTensor.h"

namespace gblas {

template<typename T>
gStatus Operations::axpy(uint64_t a, const GTensor &x, const GTensor &y, GTensor &out, bool transposeX, bool transposeY)
{
    // validate inputs;
    // perform operation
    DataBuffer &outDBuf = *out.getDataBuffer();
    const DataBuffer &xDBuf = *x.getDataBuffer();
    const DataBuffer &yDBuf = *y.getDataBuffer();
    for (unsigned i = 0; i < out.getTotalSizeInElements(); ++i)
    {
        outDBuf[i] = x * reinterpret_cast<T>(xDBuf[i]) + reinterpret_cast<T>(yDBuf[i]);
    }
    return gStatus::gBLAS_PASS;
}

} // namespace gblas