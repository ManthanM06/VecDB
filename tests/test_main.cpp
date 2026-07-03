#include <gtest/gtest.h>

#include <fstream>
#include <nlohmann/json.hpp>
#include <string>

#include "vecdb/core.hpp"

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

TEST(EngineTest, ExactBruteForceMatchesGroundTruth) {
  // 1. Load the JSON Data
  std::string filePath = std::string(TEST_DATA_DIR) + "/ground_truth.json";
  std::ifstream file(filePath);
  ASSERT_TRUE(file.is_open()) << "Failed to open: " << filePath;
  json dataset;
  file >> dataset;

  // 2. Initialize our C++ Engine (128 dimensions)
  vecdb::VectorEngine engine(128);

  // 3. Insert the 10,000 base vectors into the engine
  // We assign IDs sequentially (0 to 9999) so they perfectly match
  // the row indices returned by Scikit-Learn.
  const auto& base_vectors = dataset["base_vectors"];
  vecdb::VectorId current_id = 0;
  for (const auto& vec_data : base_vectors) {
    std::vector<float> vec = vec_data.get<std::vector<float>>();
    engine.insert(current_id++, vec);
  }

  // Sanity check
  ASSERT_EQ(engine.size(), 10000);

  // 4. Run queries and verify exact matches
  const auto& queries = dataset["queries"];
  int query_index = 0;

  for (const auto& query_obj : queries) {
    std::vector<float> query_vec =
        query_obj["query_vector"].get<std::vector<float>>();
    std::vector<vecdb::VectorId> expected_indices =
        query_obj["expected_indices"].get<std::vector<vecdb::VectorId>>();

    // Ask our C++ engine for the Top 10 closest vectors
    auto results = engine.search(query_vec, 10);

    ASSERT_EQ(results.size(), 10)
        << "Engine did not return exactly 10 results.";

    // 5. The Moment of Truth: Compare our IDs with Python's IDs
    for (size_t i = 0; i < 10; ++i) {
      EXPECT_EQ(results[i].id, expected_indices[i])
          << "Mismatch at Rank " << i << " for Query " << query_index;
    }
    query_index++;
  }
}