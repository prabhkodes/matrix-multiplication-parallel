# pragma once
#include <iostream>
#include <map>
#include <string>
#include <chrono>
#include <cstdint>
#include <vector>
#include <mpi.h>
#include <fstream>
#include <iomanip>  
#include <sstream>
#include <unordered_map>
#include <cstdlib>


struct TimeTable {
    std::string   label;
    std::uint64_t time_elapsed;
};

class ParallelTimer {
    std::string what;
    std::chrono::steady_clock::time_point start{};
    int world_rank = 0, world_size = 1;

    inline static std::vector<TimeTable> table; // shared per process

public:
    explicit ParallelTimer(const std::string& label) : what(label) {
        MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
        MPI_Comm_size(MPI_COMM_WORLD, &world_size);
        start = std::chrono::steady_clock::now();
    }
    ~ParallelTimer() {
        using namespace std::chrono;
        const auto us = duration_cast<microseconds>(steady_clock::now() - start).count();
        table.push_back({what, static_cast<std::uint64_t>(us)});
    }

static void gather_timings(const std::string& filename,
                           bool scale_avg_by_world = false,
                           MPI_Comm comm = MPI_COMM_WORLD) {
    int rank = 0, size = 1;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);

    // Serialize local results as "label\t<us>\n"
    std::ostringstream oss;
    for (const auto& t : table)
        oss << t.label << '\t' << t.time_elapsed << '\n';
    const std::string local = oss.str();
    const int nbytes = static_cast<int>(local.size());

    // Gather sizes (only root needs full vector)
    std::vector<int> counts(rank == 0 ? size : 0);
    MPI_Gather(&nbytes, 1, MPI_INT,
               rank == 0 ? counts.data() : nullptr, 1, MPI_INT,
               0, comm);

    // Root prepares displacement + recv buffer
    std::vector<int> displs;
    std::vector<char> recvbuf;
    if (rank == 0) {
        displs.resize(size);
        int total = 0;
        for (int i = 0; i < size; ++i) {
            displs[i] = total;
            total += counts[i];
        }
        recvbuf.resize(total);
    }

    // Gather all text blobs
    MPI_Gatherv(local.data(), nbytes, MPI_CHAR,
                rank == 0 ? recvbuf.data() : nullptr,
                rank == 0 ? counts.data()  : nullptr,
                rank == 0 ? displs.data()  : nullptr,
                MPI_CHAR, 0, comm);

    if (rank != 0) return;

    // --- Aggregate ---
    std::unordered_map<std::string, std::pair<long double, std::uint64_t>> agg;
    std::istringstream in(std::string(recvbuf.data(), recvbuf.size()));
    std::string line;
    while (std::getline(in, line)) {
        if (line.empty()) continue;
        const auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        const std::string label = line.substr(0, tab);
        const std::uint64_t us = std::strtoull(line.c_str() + tab + 1, nullptr, 10);
        auto& p = agg[label];
        p.first  += us;  // sum
        p.second += 1;   // count
    }

    // --- Write simple averages ---
    std::ofstream out(filename);
    out.setf(std::ios::fixed);
    out << std::setprecision(3);

    for (const auto& kv : agg) {
        long double avg_us = kv.second.first / kv.second.second;
        if (scale_avg_by_world) avg_us *= size;

        if (avg_us >= 1'000'000.0L)
            out << kv.first << '\t' << (avg_us / 1e6L) << " s\n";
        else if (avg_us >= 1'000.0L)
            out << kv.first << '\t' << (avg_us / 1e3L) << " ms\n";
        else
            out << kv.first << '\t' << static_cast<std::uint64_t>(avg_us) << " us\n";
    }
}

};