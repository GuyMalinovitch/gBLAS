#include <gtest/gtest.h>
#include "gTensor/gTensor.h"

using namespace gblas;

class GTensorTest : public testing::Test
{
public:
    void allocateTensor(TSizeArr sizes, TStrideArr strides, DType dtype)
    {
        tensor = {sizes, strides, sizes.size(), dtype};
    }
protected:
    GTensor tensor;
};

TEST_F(GTensorTest, creation_dense_bf16)
{
    allocateTensor({100, 100}, {1, 100}, DType::bf16);
    EXPECT_EQ(tensor.getRank(), 2);
    EXPECT_EQ(tensor.getTotalSizeInElements(), 10000);
    EXPECT_EQ(tensor.getTotalSizeInBytes(), 10000 * sizeof(uint16_t));
}

TEST_F(GTensorTest, creation_strided_bf16)
{
    allocateTensor({100, 100}, {1, 200}, DType::bf16);
    EXPECT_EQ(tensor.getRank(), 2);
    EXPECT_EQ(tensor.getTotalSizeInElements(), 10000);
    EXPECT_EQ(tensor.getTotalSizeInBytes(), 200*100-100 * sizeof(uint16_t));
}

TEST_F(GTensorTest, data_injection_bf16)
{
    allocateTensor({100, 100}, {1, 100}, DType::int32);
    auto dataArr = new std::array<uint32_t, 100*100>;
    std::fill(dataArr->begin(), dataArr->end(), 1);
    tensor.initData(dataArr);
    auto val = (*tensor.getDataBuffer())[100];
    EXPECT_EQ(*(int32_t*)(val), 1);
}