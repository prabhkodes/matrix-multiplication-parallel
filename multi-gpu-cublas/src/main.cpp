#include <thread>


#include "../include/parallel_2.hpp"
#include "../include/CMatrix.hpp"



int main(int argc, char ** argv) {


    MPI_Init(&argc, &argv);
    int world_rank, world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    int deviceCount = 0;
    cudaGetDeviceCount(&deviceCount);
    cudaSetDevice(world_rank % deviceCount);

    if (world_rank == 0) {
            { 
                using namespace std::chrono; 
                auto n=system_clock::now(); 
                auto ms=duration_cast<milliseconds>(n.time_since_epoch())%1000; 
                std::time_t tt=system_clock::to_time_t(n); 
                
                std::cout << "STARTING execution at rank 0"  << " || "
                        << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S") << '.'
                        << std::setw(3) << std::setfill('0') << ms.count() // Sets fill to '0'
                        << '\n'; 
                
                
                std::cout << std::setfill(' '); 
            }
        }

    long int N{10000};

    // Not dealing with rem right now
    if (N % world_size != 0) {
        if (world_rank == 0) {
            std::cerr << "Error: N (" << N << ") must be divisible by world_size ("
                      << world_size << ").\n";
        }
        MPI_Abort(MPI_COMM_WORLD, 1);
    } // Not dealing with rem right now

    long int n_loc = N /world_size;

    CMatrix<double> A(n_loc, N);
    CMatrix<double> B(n_loc, N);

    {
    CTimer t("INIT A,B");
    A.fill_rand();
    B.fill_identity();
    }

    A.print_mat_with_label("This is A: ");
    B.print_mat_with_label("This is B: ");

    auto C = A * B;

    C.print_mat_with_label("This is C: ");

    std::vector<TimerData> all_timings;
    CTimer::gather_and_print(0, all_timings);


    if (world_rank == 0) {
            { 
                using namespace std::chrono; 
                auto n=system_clock::now(); 
                auto ms=duration_cast<milliseconds>(n.time_since_epoch())%1000; 
                std::time_t tt=system_clock::to_time_t(n); 
                
                std::cout << "ENDING execution at rank 0"  << " || "
                        << std::put_time(std::localtime(&tt), "%Y-%m-%d %H:%M:%S") << '.'
                        << std::setw(3) << std::setfill('0') << ms.count() // Sets fill to '0'
                        << '\n'; 
                
                
                std::cout << std::setfill(' '); 
            }
        }

    MPI_Finalize();


    return 0;

}