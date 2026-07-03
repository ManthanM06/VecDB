#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>  // Add string header

using json = nlohmann::json;

TEST(GroundTruthTest, CanLoadAndVerifyJsonData) {
  // Construct the absolute path using the macro from CMake
  std::string filePath = std::string(TEST_DATA_DIR) + "/ground_truth.json";
  std::ifstream file(filePath);

  // Improved error message to show exactly where it tried to look
  ASSERT_TRUE(file.is_open()) << "Failed to open: " << filePath;

  // Parse the JSON file
  json dataset;
  file >> dataset;

  // 1. Verify the core keys exist
  ASSERT_TRUE(dataset.contains("base_vectors"));
  ASSERT_TRUE(dataset.contains("queries"));

  // 2. Verify the dataset sizes match our Python script (10,000 vectors, 100
  // queries)
  EXPECT_EQ(dataset["base_vectors"].size(), 10000);
  EXPECT_EQ(dataset["queries"].size(), 100);

  // 3. Verify the dimensionality is exactly 128
  EXPECT_EQ(dataset["base_vectors"][0].size(), 128);
  EXPECT_EQ(dataset["queries"][0]["query_vector"].size(), 128);

  // 4. Verify that each query expects exactly 10 nearest neighbors (Top-K)
  EXPECT_EQ(dataset["queries"][0]["expected_indices"].size(), 10);
}