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

TEST(PersistenceTest, CanSaveAndLoadBinaryDatabase) {
  std::string db_path = "test_database.vec";

  // force nukde file fome prev test
  std::remove(db_path.c_str());

  // Create a scope to force the engine to be destroyed
  {
    vecdb::VectorEngine engine(128);

    // Insert a few dummy vectors
    // vector 1 (1 on the first dim , 0 everywhere else)
    std::vector<float> vec1(128, 0.0f);
    vec1[0] = 1.0f;
    // vector 2 (1 on the first dim , 0 everywhere else)
    std::vector<float> vec2(128, 0.0f);  // Vector of 128 twos
    vec2[1] = 1.0f;

    engine.insert(100, vec1);
    engine.insert(200, vec2);

    // Save to disk
    EXPECT_NO_THROW(engine.save(db_path));
  }  // 'engine' is completely destroyed here, RAM is cleared.

  // Spin up a brand new, empty engine
  vecdb::VectorEngine loaded_engine(128);

  // Load the binary file from disk
  EXPECT_NO_THROW(loaded_engine.load(db_path));

  // Verify the state perfectly matches
  ASSERT_EQ(loaded_engine.size(), 2);

  // Query it to prove the math still works on the loaded data
  std::vector<float> query(128, 0.0f);
  query[0] = 1.0f;
  auto results = loaded_engine.search(query, 2);

  ASSERT_EQ(results.size(), 2);

  // --- DIAGNOSTIC OUTPUT ---
  // std::cout << "\n[--- ENGINE CONFESSION ---]\n";
  // std::cout << "Result 0: ID = " << results[0].id
  //           << " | Distance = " << results[0].distance << "\n";
  // std::cout << "Result 1: ID = " << results[1].id
  //           << " | Distance = " << results[1].distance << "\n";
  // std::cout << "[-------------------------]\n\n";
  // -------------------------

  EXPECT_EQ(results[0].id, 100);
  EXPECT_EQ(results[1].id, 200);
}