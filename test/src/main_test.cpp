#include <gtest/gtest.h>

#include "nmbuflowtorch/definitions.hpp"
#include "nmbuflowtorch/layer.hpp"
#include "nmbuflowtorch/layer/dense.hpp"
#include "nmbuflowtorch/layer/sigmoid.hpp"

TEST(DenseTest, InitializeDenseLayer)
{
  nmbuflowtorch::Layer* dense1 = new nmbuflowtorch::layer::Dense(2, 2);
  EXPECT_EQ(dense1->get_parameters().size(), 5);
}

int main(int argc, char** argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
