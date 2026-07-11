#include "vecdb/core.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

using json = nlohmann::json;

// Global database instance
vecdb::VectorEngine db(128, 16); 
const std::string DB_FILE = "production_database.vec";

int main() {
    httplib::Server svr;

    // Attempt to load existing database on startup
    try {
        db.load(DB_FILE);
        std::cout << "[INFO] Loaded existing database with " << db.size() << " vectors.\n";
    } catch (const std::exception& e) {
        std::cout << "[INFO] Starting with fresh database. (" << e.what() << ")\n";
    }

    // ---------------------------------------------------------
    // ENDPOINT 1: INSERT
    // Expects JSON: { "id": 123, "vector": [0.1, 0.2, ... 128 floats] }
    // ---------------------------------------------------------
    svr.Post("/insert", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            vecdb::VectorId id = j["id"].get<vecdb::VectorId>();
            std::vector<float> data = j["vector"].get<std::vector<float>>();

            db.insert(id, data);
            
            res.status = 200;
            res.set_content(R"({"status":"success"})", "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(R"({"error":")" + std::string(e.what()) + R"("})", "application/json");
        }
    });

    // ---------------------------------------------------------
    // ENDPOINT 2: SEARCH
    // Expects JSON: { "k": 10, "vector": [0.1, 0.2, ... 128 floats] }
    // ---------------------------------------------------------
    svr.Post("/search", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            size_t k = static_cast<size_t>(j.value("k", 10)); // Default to Top-10 if not provided
            std::vector<float> query = j["vector"].get<std::vector<float>>();

            auto results = db.search(query, k);

            // Construct JSON response
            json response_json = json::array();
            for (const auto& r : results) {
                response_json.push_back({
                    {"id", r.id},
                    {"distance", r.distance}
                });
            }
            
            res.status = 200;
            res.set_content(response_json.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(R"({"error":")" + std::string(e.what()) + R"("})", "application/json");
        }
    });

    // ---------------------------------------------------------
    // ENDPOINT 3: SAVE
    // Triggers a disk persistence event
    // ---------------------------------------------------------
    svr.Post("/save", [](const httplib::Request& req, httplib::Response& res) {
        try {
            db.save(DB_FILE);
            res.status = 200;
            res.set_content(R"({"status":"saved"})", "application/json");
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(R"({"error":")" + std::string(e.what()) + R"("})", "application/json");
        }
    });

    std::cout << "[INFO] VecDB Server listening on http://localhost:8080\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}