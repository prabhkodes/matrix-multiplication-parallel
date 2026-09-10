#include <iostream>
#include <cmath>
#include <mpi.h>

#include "../include/parallel_2.hpp"
#include "../include/CMatrix.hpp"

inline void log_ts(const char* msg) {
    using namespace std::chrono;
    auto n = system_clock::now();
    auto ms = duration_cast<milliseconds>(n.time_since_epoch()) % 1000;
    std::time_t tt = system_clock::to_time_t(n);
    std::cout << msg << " || "
              << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
              << '.' << std::setw(3) << std::setfill('0') << ms.count()
              << std::endl;
}

int main(int argc, char** argv) {
    MPI_Init(&argc, &argv);

    int world_rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    if (world_rank == 0) log_ts("Starting Cannon Matrix Multiplication");

    long N = 10000; // global matrix size
    int q = static_cast<int>(std::sqrt(world_size));

    if (q * q != world_size) {
        if (world_rank == 0)
            std::cerr << "Error: world_size must be a perfect square for Cannon algorithm.\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    if (N % q != 0) {
        if (world_rank == 0)
            std::cerr << "Error: N must be divisible by sqrt(world_size).\n";
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long n = N / q; // local tile size


    int dims[2] = { q, q };
    int periods[2] = { 1, 1 }; // wrap-around (torus)
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    CMatrix<double> A(n, cart_comm, N, q);
    CMatrix<double> B(n, cart_comm, N, q);

    {    CTimer t("INIT A,B");
        #pragma omp parallel sections
        {
            #pragma omp section
            {A.fill_rand(); }

            #pragma omp section
            {B.fill_identity(); }
        }
    } 
    auto C = A * B;

    std::vector<TimerData> all_timings;
    CTimer::gather_and_print(0, all_timings);

    if (world_rank == 0) log_ts("Finished Cannon Matrix Multiplication");

    MPI_Finalize();
    return 0;
}
