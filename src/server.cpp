#include "vecdb/core.hpp"
#include <httplib.h>
#include <nlohmann/json.hpp>
#include <iostream>
#include <string>

using json = nlohmann::json;

// Global database instance — 384 dimensions to match all-MiniLM-L6-v2
vecdb::VectorEngine db(384, 16);
const std::string DB_FILE = "production_database.vec";

// ---------------------------------------------------------------------------
// Helper: build a well-formed JSON error response
// ---------------------------------------------------------------------------
static std::string json_error(const std::string& msg) {
    return json{{"error", msg}}.dump();
}

int main() {
    httplib::Server svr;

    // Attempt to load existing database on startup
    try {
        db.load(DB_FILE);
        std::cout << "[INFO] Loaded existing database. "
                  << "Live: " << db.size()
                  << "  Tombstoned: " << db.deleted_count() << " vectors.\n";
    } catch (const std::exception& e) {
        std::cout << "[INFO] Starting with fresh database. ("
                  << e.what() << ")\n";
    }

    // =========================================================================
    // ENDPOINT 1: INSERT
    // Body: { "id": 123, "vector": [...], "metadata": {"key": "val"} }
    //   metadata field is optional.
    // =========================================================================
    svr.Post("/insert", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);

            vecdb::VectorId      id   = j["id"].get<vecdb::VectorId>();
            std::vector<float>   data = j["vector"].get<std::vector<float>>();
            nlohmann::json       meta = j.value("metadata", nlohmann::json{});

            db.insert(id, data, meta);

            res.status = 200;
            res.set_content(R"({"status":"success"})", "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    // =========================================================================
    // ENDPOINT 2: SEARCH
    // Body: {
    //   "k": 10,
    //   "vector": [...],
    //   "filter": { "key": "author", "value": "alice" }   <- optional
    // }
    // Returns: [ { "id": 1, "distance": 0.05, "metadata": {...} }, ... ]
    // =========================================================================
    svr.Post("/search", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);

            size_t             k     = static_cast<size_t>(j.value("k", 10));
            std::vector<float> query = j["vector"].get<std::vector<float>>();

            // Build optional metadata filter
            vecdb::MetadataFilter filter;
            if (j.contains("filter") && !j["filter"].is_null()) {
                filter.key    = j["filter"]["key"].get<std::string>();
                filter.value  = j["filter"]["value"];
                filter.active = true;
            }

            auto results = db.search(query, k, filter);

            json response_arr = json::array();
            for (const auto& r : results) {
                json entry = {{"id", r.id}, {"distance", r.distance}};
                if (!r.metadata.is_null() && !r.metadata.empty()) {
                    entry["metadata"] = r.metadata;
                }
                response_arr.push_back(entry);
            }

            res.status = 200;
            res.set_content(response_arr.dump(), "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    // =========================================================================
    // ENDPOINT 3: DELETE (soft-delete / tombstone)
    // Body: { "id": 123 }
    // =========================================================================
    svr.Post("/delete", [](const httplib::Request& req, httplib::Response& res) {
        try {
            auto j = json::parse(req.body);
            vecdb::VectorId id = j["id"].get<vecdb::VectorId>();

            db.remove(id);

            res.status = 200;
            res.set_content(
                json{{"status", "deleted"}, {"id", id}}.dump(),
                "application/json");
        } catch (const std::exception& e) {
            res.status = 400;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    // =========================================================================
    // ENDPOINT 4: SAVE (async — non-blocking)
    // Triggers a background snapshot + disk write.
    // Returns 202 if save kicked off, 409 if one is already in progress.
    // =========================================================================
    svr.Post("/save", []([[maybe_unused]] const httplib::Request& req, httplib::Response& res) {
        try {
            bool started = db.save_async(DB_FILE);
            if (started) {
                res.status = 202;
                res.set_content(
                    R"({"status":"saving_async","message":"Background save initiated."})",
                    "application/json");
            } else {
                res.status = 409;
                res.set_content(
                    R"({"status":"busy","message":"A save is already in progress."})",
                    "application/json");
            }
        } catch (const std::exception& e) {
            res.status = 500;
            res.set_content(json_error(e.what()), "application/json");
        }
    });

    // =========================================================================
    // ENDPOINT 5: STATUS (GET)
    // Returns live counts and whether a background save is running.
    // =========================================================================
    svr.Get("/status", []([[maybe_unused]] const httplib::Request& req, httplib::Response& res) {
        json status = {
            {"live_vectors",      static_cast<size_t>(db.size())},
            {"total_vectors",     static_cast<size_t>(db.total_size())},
            {"deleted_vectors",   static_cast<size_t>(db.deleted_count())},
            {"save_in_progress",  db.is_save_in_progress()},
            {"dimensions",        static_cast<size_t>(db.dimensions())}
        };
        res.status = 200;
        res.set_content(status.dump(), "application/json");
    });

    // =========================================================================
    // ENDPOINT 6: INFO (GET) — legacy alias for /status
    // =========================================================================
    svr.Get("/info", []([[maybe_unused]] const httplib::Request& req, httplib::Response& res) {
        json info = {
            {"live",    static_cast<size_t>(db.size())},
            {"deleted", static_cast<size_t>(db.deleted_count())},
            {"total",   static_cast<size_t>(db.total_size())}
        };
        res.status = 200;
        res.set_content(info.dump(), "application/json");
    });

    std::cout << "[INFO] VecDB Server (Phase 7) listening on http://localhost:8080\n";
    std::cout << "[INFO] Endpoints: /insert  /search  /delete  /save  /status  /info\n";
    svr.listen("0.0.0.0", 8080);

    return 0;
}