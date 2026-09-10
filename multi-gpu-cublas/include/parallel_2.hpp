
#ifndef PARALLEL_TIMER_HPP
#define PARALLEL_TIMER_HPP

/**
*  USE THIS AS
*  {CTimer t("WHATEVER YOU ARE TIMING");
*        THE CODE YOU ARE TIMING
*    }
*
*  THEN AT THE END OF MAIN PUT
*  CTimer::print_timing_results();
*  Note that this header rquires -std=c++20 and -lmpi
*/
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <iostream>
#include <map>
#include <ostream>
#include <string>
#include <mpi.h>
#include <numeric>
#include <ranges>
#include <vector>
#include <fstream>

struct TimerData
{
    int rank{0};
    long total_time{0};
    int calls{0};
};

static std::map<std::string, TimerData> times{};

class CTimer
{
public:
    std::chrono::steady_clock::time_point start, end;
    std::string timewhat;

    //Constructor
    explicit CTimer( const std::string & function);

    //Destructor
    ~CTimer();

    //Utilities
    static void print_timing_results();
    static void gather_and_print(int root, std::vector<TimerData>& all_timings);
    static void print_statistics(const std::vector<TimerData>& all_timings);

};

inline CTimer::CTimer(const std::string& function)
{
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    timewhat = function;
    if (!times.contains(timewhat))
        times.insert(std::pair<std::string, TimerData>(timewhat, {rank,0,0}));
    start = std::chrono::steady_clock::now();
}

inline CTimer::~CTimer()
{
    end = std::chrono::steady_clock::now();
    const auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    times.at(timewhat).total_time += duration;
    times.at(timewhat).calls++;
}

inline void CTimer::print_timing_results()
{
    for (const auto& [fst, snd] : times)
        std::cout  << "Function: " << std::setw(15) << std::left << fst  << " Time (ms): " <<  std::setw(8) << std::left  << snd.total_time <<" Total calls: " << snd.calls << std::endl;
}

inline void CTimer::gather_and_print(const int root, std::vector<TimerData>& all_timings)
{
    const size_t times_size = times.size();

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    if (rank == root)
        all_timings.resize(world_size * times_size);

    //Create an array of timerdata to be gathered on root
    std::vector<TimerData> timings(times_size);

    int counter = 0;
    for (const auto& snd : times | std::views::values)
    {
        timings[counter] = snd;
        counter++;
    }

    //Gather data on root
    MPI_Gather(timings.data(), sizeof(TimerData) * times.size(), MPI_BYTE, all_timings.data(), sizeof(TimerData) * times.size(), MPI_BYTE, root, MPI_COMM_WORLD );

    if (rank == 0)
        CTimer::print_statistics(all_timings);
    //Assuming all processes call the same function there is no need to send the functions names themselves
    //all_timings is a vector container where each block in the range times.size() * i and times.size()* ( i + 1)
    // contains the times of the i-th process
}

inline void CTimer::print_statistics(const std::vector<TimerData>& all_timings)
{
    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    std::ofstream file("statistics.txt");

    // Total number of functions
    const size_t functions = times.size();
    int counter = 0;
    for (const auto& key : times | std::views::keys)
    {file << std::string(100, '-') << std::endl;
     file << "Function : " << key << std::endl;
     for (int i = 0; i < world_size; i++)
     {
         file << "Rank " << all_timings[counter + i*functions].rank << " process time : "  << all_timings[counter + i * functions].total_time << " ms" << std::endl;
     }
     counter++;
    }
    file << std::string(100, '-') << std::endl;
    std::vector<long> compare(world_size);

    // Collecting the times for each function and then printing max, min and avg
    counter = 0;
    for (const auto& key : times | std::views::keys)
    {

        for (int i = 0; i < world_size; i++)
        {
            compare[i] = all_timings[counter  + i * functions].total_time;
        }
        counter++;
        const long max_time = *std::ranges::max_element(compare);
        const long min_time = *std::ranges::min_element(compare);
        const double avg_time = std::accumulate(compare.begin(), compare.end(), 0.0) / world_size;
        file << std::string(100, '-') << std::endl;
        file << "Function : " << std::setw(20) << std::left << key  << " Max time: " << std::setw(5) << std::right << max_time << "ms Min time: "
        << std::setw(5) << std::right << min_time << "ms Avg time : " << std::setw(5) << std::right << avg_time  << "ms" << std::endl;
    }
    file << std::string(100, '-') << std::endl;
    file.close();

}

#endif //PARALLEL_TIMER_HPP