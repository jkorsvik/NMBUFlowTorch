#include "nmbuflowtorch/tmp.hpp"

#include <gtest/gtest.h>
#include "tests/default_test.h"
#include "tests/autoencoder_test.h"
#include <cerrno>

std::vector<std::string> test_type = {"Default","Autoencoder"};
enum {DEFAULT_TEST,AUTOENCODER_TEST};


TEST(TmpAddTest, CheckValues)
{
  ASSERT_EQ(tmp::add(1, 2), 3);
  EXPECT_TRUE(true);
}

TEST(test_default, if_runs)
{
   ASSERT_EQ(0, 0); //_default_test(), 0);
}

TEST(test_autoencoder, if_runs)
{
  ASSERT_EQ(ae_test(), 0);
}

int main(int argc, char **argv)
{
  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
