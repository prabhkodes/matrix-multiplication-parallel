#include <omp.h>        

#include "../include/parallel_2.hpp"
#include "../include/CMatrix.hpp"

inline void log_ts(const char* msg) {
    using namespace std::chrono;
    auto old_fill = std::cout.fill();          
    auto n  = system_clock::now();
    auto ms = duration_cast<milliseconds>(n.time_since_epoch()) % 1000;
    std::time_t tt = system_clock::to_time_t(n);

    std::cout << msg << " || "
              << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S")
              << '.' << std::setw(3) << std::setfill('0') << ms.count()
              << '\n';

    std::cout.fill(old_fill);                  
}


int main(int argc, char ** argv) {
    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if (world_rank==0) log_ts("Starting execution at rank 0");

    long int N{10000};
    if (N % world_size != 0) {
        if (world_rank == 0) {
            std::cerr << "Error: N (" << N << ") must be divisible by world_size ("
                      << world_size << ").\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    }

    long int n_loc = N / world_size;

    CMatrix<double> A(n_loc, N);
    CMatrix<double> B(n_loc, N);

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


    if (world_rank==0) log_ts("Ending execution at rank 0");

    std::vector<TimerData> all_timings;
    CTimer::gather_and_print(0, all_timings);

    MPI_Finalize();

    return 0;
}
