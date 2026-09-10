#pragma once
#include <vector>
#include <iostream>
#include <iomanip>
#include <cassert>
#include <mpi.h>
#include <cblas.h>

template <typename T>
class CMatrix {
public:
    std::vector<T> data;    // local n×n tile
    std::vector<T> sub_mat; // temp copy if needed (used in extract_block)

    long N1, N2;      // tile dimensions (both = n_tile)
    long n_tile;      // local tile size
    long N_global;    // full global matrix size

    int world_rank, world_size;
    int q;            // sqrt(P)
    int coords[2];    // (row, col) position in process grid

    MPI_Comm cart_comm;    // 2D Cartesian communicator
    int nbr_left, nbr_right;
    int nbr_up, nbr_down;

    // ---- Constructor ----
    CMatrix(long n_tile_in, MPI_Comm cart, long N_global_in, int q_in);

    // ---- Fillers ----
    void fill_rand();      
    void fill_identity();  

    // ---- Helpers ----
    void extract_block(int iter);  
    void print_mat_with_label(const std::string& label) const;

    // ---- Operator overload ----
    template<typename M>
    friend CMatrix<M> operator*(const CMatrix<M>& A, const CMatrix<M>& B);
};




template <typename T>
void CMatrix<T>::extract_block(int /*iter*/) {
    // In Cannon, each rank already holds its working n×n tile.
    // Just mirror it into sub_mat for a uniform call site.
    sub_mat = data;  // O(n^2) copy
}


// Constructor
template<typename T>
CMatrix<T>::CMatrix(long n_tile_in, MPI_Comm cart, long N_global_in, int q_in)
: N1(n_tile_in),
  N2(n_tile_in),
  n_tile(n_tile_in),
  N_global(N_global_in),
  q(q_in),
  cart_comm(cart)
{
    MPI_Comm_rank(cart_comm, &world_rank);
    MPI_Comm_size(cart_comm, &world_size);

    MPI_Cart_coords(cart_comm, world_rank, 2, coords);

    MPI_Cart_shift(cart_comm, 1, 1, &nbr_left,  &nbr_right);
    MPI_Cart_shift(cart_comm, 0, 1, &nbr_up,    &nbr_down);

    data.resize(N1 * N2);
}
// Constructor



template <typename T>
void CMatrix<T>::fill_rand() {
    std::srand(static_cast<unsigned>(std::time(nullptr)) + world_rank);

    for (long i = 0; i < N1; ++i) {
        for (long j = 0; j < N2; ++j) {
            data[i * N2 + j] = static_cast<T>(1 + std::rand() % 9); // values 1–9
        }
    }
}




// Diag elems 1 and rest 0 for matrix.data
template <typename T>
void CMatrix<T>::fill_identity() {
    std::fill(data.begin(), data.end(), T{0});

    if (coords[0] == coords[1]) {
        for (long i = 0; i < N1; ++i) {
            data[i * N2 + i] = T{1};
        }
    }
}
// Diag elems 1 and rest 0 for matrix.data



// Print matrix.data
template <typename T>
void CMatrix<T>::print_mat_with_label(const std::string& label) const {
    MPI_Barrier(cart_comm); // sync all ranks

    for (int r = 0; r < world_size; ++r) {
        if (world_rank == r) {
            std::cout << "\n==========[ Rank " << world_rank 
                      << " | coords(" << coords[0] << "," << coords[1] << ") ]==========\n";
            std::cout << ">> " << label 
                      << " (tile " << N1 << " x " << N2 << ")\n";

            std::cout << std::fixed << std::setprecision(2);
            for (long i = 0; i < N1; ++i) {
                std::cout << "| ";
                for (long j = 0; j < N2; ++j) {
                    std::cout << std::setw(6) << data[i * N2 + j] << " ";
                }
                std::cout << "|\n";
            }
            std::cout.flush();
        }
        MPI_Barrier(cart_comm); // let ranks print one by one
    }
}
// Print matrix.data



// OPERATOR OVERLOADS

template <typename M>
CMatrix<M> operator*(const CMatrix<M>& A, const CMatrix<M>& B) {
    assert(A.q == B.q && A.N1 == B.N1 && A.N2 == B.N2);
    assert(A.cart_comm != MPI_COMM_NULL && B.cart_comm != MPI_COMM_NULL);

    const int q = A.q;
    const long n = A.n_tile;

    CMatrix<M> C(n, A.cart_comm, A.N_global, A.q);
    std::fill(C.data.begin(), C.data.end(), M{0});

    std::vector<M> A_buf = A.data;
    std::vector<M> B_buf = B.data;

    // ---- Initial skew ----
    for (int k = 0; k < A.coords[0]; ++k)
        MPI_Sendrecv_replace(A_buf.data(), n*n, MPI_DOUBLE,
                             A.nbr_left, 0, A.nbr_right, 0,
                             A.cart_comm, MPI_STATUS_IGNORE);

    for (int k = 0; k < A.coords[1]; ++k)
        MPI_Sendrecv_replace(B_buf.data(), n*n, MPI_DOUBLE,
                             A.nbr_up, 0, A.nbr_down, 0,
                             A.cart_comm, MPI_STATUS_IGNORE);

    // ---- Cannon main loop ----
    for (int step = 0; step < q; ++step) {
        {
            CTimer t("DGEMM");
            cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                        n, n, n, 1.0,
                        A_buf.data(), n,
                        B_buf.data(), n,
                        1.0, C.data.data(), n);
        }

        // shift A left, B up
        if (step < q - 1) {
            CTimer t("Shift A,B");
            MPI_Sendrecv_replace(A_buf.data(), n*n, MPI_DOUBLE,
                                 A.nbr_left, 1, A.nbr_right, 1,
                                 A.cart_comm, MPI_STATUS_IGNORE);
            MPI_Sendrecv_replace(B_buf.data(), n*n, MPI_DOUBLE,
                                 A.nbr_up, 2, A.nbr_down, 2,
                                 A.cart_comm, MPI_STATUS_IGNORE);
        }
    }

    return C;
}
