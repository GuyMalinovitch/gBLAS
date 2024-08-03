#include <gtest/gtest.h>
#include "gTensor/gTensor.h"

using namespace gblas;

class GTensorTest : public testing::Test
{
public:
    void allocateTensor(TSizeArr sizes, TStrideArr strides, unsigned rank, DType dtype)
    {
        tensor = gTensor{sizes, strides, rank, dtype};
    }
protected:
    gTensor tensor;
};

TEST_F(GTensorTest, creation_dense_bf16)
{
    allocateTensor({100, 100, 1, 1, 1}, {1, 100, 10000, 10000, 10000}, 5, DType::bf16);
    EXPECT_EQ(tensor.getRank(), 5);
    EXPECT_EQ(tensor.getTotalSizeInElements(), 10000);
    EXPECT_EQ(tensor.getMemorySizeInBytes(), 10000 * sizeof(uint16_t));
}

TEST_F(GTensorTest, creation_strided_bf16)
{
    allocateTensor({100, 100, 1, 1, 1}, {1, 200, 40000, 40000, 40000}, 5, DType::bf16);
    EXPECT_EQ(tensor.getRank(), 5);
    EXPECT_EQ(tensor.getTotalSizeInElements(), 10000);
    EXPECT_EQ(tensor.getMemorySizeInBytes(), (200*99+100) * sizeof(uint16_t));
}

TEST_F(GTensorTest, data_injection_int32)
{
    allocateTensor({100, 100, 1, 1, 1}, {1, 100, 10000, 10000, 10000}, 5, DType::int32);
    auto dataArr = new std::array<uint32_t, 100*100>;
    std::fill(dataArr->begin(), dataArr->end(), 1);
    tensor.initData(dataArr);
    auto val = (*tensor.getDataBuffer())[100];
    EXPECT_EQ(*(int32_t*)(val), 1);
}

TEST_F(GTensorTest, data_extraction_int32)
{
    allocateTensor({100, 100, 1, 1, 1}, {1, 100, 10000, 10000, 10000}, 5, DType::int32);
    auto dataArr = new std::array<uint32_t, 100*100>;
    unsigned value = 0;
    std::generate(dataArr->begin(), dataArr->end(), [&value](){return value++;});
    tensor.initData(dataArr);

    for(unsigned i = 0; i < tensor.getTotalSizeInElements(); i++)
    {
        EXPECT_EQ(*(int32_t*)(tensor[i]), i);
    }
}

TEST_F(GTensorTest, data_extraction_coordinates_int32)
{
    allocateTensor({100, 100, 1, 1, 1}, {1, 100, 10000, 10000, 10000}, 5, DType::int32);
    auto dataArr = new std::array<uint32_t, 100*100>;
    unsigned value = 0;
    std::generate(dataArr->begin(), dataArr->end(), [&value](){return value++;});
    tensor.initData(dataArr);
    Coordinates coords = {0, 0};
    EXPECT_EQ(*(int32_t*)(tensor[coords]), 0);
    coords = {50, 0};
    EXPECT_EQ(*(int32_t*)(tensor[coords]), 50);
    coords = {25, 25};
    EXPECT_EQ(*(int32_t*)(tensor[coords]), 2525);
}

TEST_F(GTensorTest, data_extraction_coordinates_strided_int32)
{
    allocateTensor({100, 100, 1, 1, 1}, {1, 200, 20000, 20000, 20000}, 5, DType::int32);
    auto dataArr = new std::array<uint32_t, 100*200>;
    unsigned value = 0;
    std::generate(dataArr->begin(), dataArr->end(), [&value](){return value++;});
    tensor.initData(dataArr);
    Coordinates coords = {0, 10};
    EXPECT_EQ(*(int32_t*)(tensor[coords]), 0*1 + 10*200);
    coords = {50, 1};
    EXPECT_EQ(*(int32_t*)(tensor[coords]), 50*1 + 1*200);
    coords = {25, 25};
    EXPECT_EQ(*(int32_t*)(tensor[coords]), 25*1 + 25*200);
}
