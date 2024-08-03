#include "data_types/non_conventional_dtypes.h"
#include <gtest/gtest.h>
#include <concepts>

// make sure the new data types implement the required methods for testing.
template<typename T>
concept NonConventionalDtype = requires(T t)
{
    {t.toFloat()}   -> std::same_as<float>;
    {t < t}         -> std::same_as<bool>;
    {T{float{}}}    -> std::same_as<T>;
    {t.isInf()}     -> std::same_as<bool>;
    {t.isZero()}    -> std::same_as<bool>;
};

template <typename T> requires NonConventionalDtype<T>
class DTypesTest : public ::testing::Test
{
public:
    DTypesTest() = default;
    virtual ~DTypesTest() = default;
};

using types = ::testing::Types<gblas::bf16_t, gblas::fp16_t, gblas::fp8_152, gblas::fp8_143>;
TYPED_TEST_SUITE(DTypesTest, types);

TYPED_TEST(DTypesTest, construction)
{
    TypeParam val(10.0f);
    EXPECT_EQ(val.toFloat(), 10.0f);
    EXPECT_FALSE(val.isNan());
    EXPECT_FALSE(val.isInf());

    val = {0.0f};
    EXPECT_EQ(val.toFloat(), 0);
    EXPECT_TRUE(val.isZero());
    val = {std::numeric_limits<float>::infinity()};
    EXPECT_TRUE(val.isInf());
    val = {std::numeric_limits<float>::quiet_NaN()};
    EXPECT_TRUE(val.isNan());
    val = {std::numeric_limits<float>::signaling_NaN()};
    EXPECT_TRUE(val.isNan());
}

