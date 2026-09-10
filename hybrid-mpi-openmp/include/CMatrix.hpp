#pragma once
#include <algorithm> 
#include <iostream>
#include <vector>
#include <cstdlib>   
#include <ctime>     
#include <cblas.h> 
#include <mpi.h>
#include <iomanip> // has set precision
#include <cassert> // has assertions

// add once in your translation unit / header:
#include <omp.h>
#include <random>

#include "../include/parallel_2.hpp"


template <typename T>
class CMatrix{
public:
    std::vector<T> data; //Actual matrix data
    std::vector<T> sub_mat; // sub matrix 

    long int N1, N2; // N1 -> no. of rows, N2 -> no. of columns
    long int n_loc;

    int world_rank, world_size; // MPI world rank and size upon init of CMatrix
    
    CMatrix(long int N1, long int N2); // Ctor
    
    void fill_rand(); // fills random T values
    void fill_identity(); // Makes diagnol elements 1 

    void print_mat_with_label(const std::string& label) const;
    void print_submat_with_label(const std::string& label) const;

    void extract_block(int iter);

    template<typename M>
    friend CMatrix<M> operator* (const CMatrix<M>& m1, const CMatrix<M>& m2);
};


template <typename T>
void CMatrix<T>::extract_block(int iter) {
    const long int col_start_offset = iter * n_loc;

    assert((col_start_offset + n_loc) <= N2);  
   
    sub_mat.resize(n_loc * n_loc);  

    #pragma omp parallel for collapse(2) schedule(static)
    for (long int i = 0; i < n_loc; ++i) {
        for (long int j = 0; j < n_loc; ++j) {
            sub_mat[i * n_loc + j] = data[i * N2 + (col_start_offset + j)];
        }
    }
}




// Constructor
template<typename T>
CMatrix<T>::CMatrix(long int n1, long int n2)  : N1(n1), N2(n2)
{
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    // Only compute n_loc if this is a global matrix
    if (N1 % world_size == 0 && N1 > world_size) {
        n_loc = N1 / world_size;
    } else {
        n_loc = N1;  // already a local matrix
    }

    data.resize(N1*N2); 
} // Constructor




// Fill random T values in matrix.data vector
template <typename T>
void CMatrix<T>::fill_rand() {
    std::srand(static_cast<unsigned>(std::time(nullptr)) + world_rank); // per-rank seed

    const T min_val = static_cast<T>(1.01);
    const T max_val = static_cast<T>(9.99);
    const T range   = max_val - min_val;

    for (long i = 0; i < N1; ++i) {
        for (long j = 0; j < N2; ++j) {
            T r = static_cast<T>(std::rand()) / static_cast<T>(RAND_MAX); 
            data[i * N2 + j] = min_val + r * range; // scale to [1.01, 9.99)
        }
    }
} // Fill random T values in matrix.data vector


// Diag elems 1 and rest 0 for matrix.data
template <typename T>
void CMatrix<T>::fill_identity() {
    std::fill(data.begin(), data.end(), T{0.0});

    const long int global_row_offset = world_rank * N1;  // N1 == n_loc

    for (long int i = 0; i < N1; ++i) {
        long int global_row = global_row_offset + i;
        if (global_row < N2) {  // valid only if inside global square matrix
            data[i * N2 + global_row] = T{1.0};
        }
    }
} // Diag elems 1 and rest 0 for matrix.data





// Print matrix.data
template <typename T>
void CMatrix<T>::print_mat_with_label(const std::string& label) const {
    for (int rank = 0; rank < world_size; ++rank) {
        MPI_Barrier(MPI_COMM_WORLD);  // Synchronize before each rank prints

        if (world_rank == rank) {
            std::cout << "\n==========[ Rank " << world_rank << " ]==========\n";
            std::cout << ">> " << label << " (dimensions: " << N1 << " x " << N2 << ")\n";
            std::cout << std::fixed << std::setprecision(3);

            for (int i = 0; i < N1; ++i) {
                std::cout << "| ";
                for (int j = 0; j < N2; ++j) {
                    std::cout << std::setw(7) << data[i * N2 + j] << " ";
                }
                std::cout << "|\n";
            }
            std::cout << std::flush;
        }

        MPI_Barrier(MPI_COMM_WORLD);  // Wait for printing to complete before next rank
    }
}
// Print matrix.data



// Print matrix.sub_mat
template <typename T>
void CMatrix<T>::print_submat_with_label(const std::string& label) const {
    for (int rank = 0; rank < world_size; ++rank) {
        MPI_Barrier(MPI_COMM_WORLD);  // Sync before each rank prints

        if (world_rank == rank) {
            std::cout << "\n==========[ Rank " << world_rank << " ]==========\n";
            std::cout << ">> " << label << " (dimensions: " << n_loc << " x " << n_loc << ")\n";
            std::cout << std::fixed << std::setprecision(3);

            for (long i = 0; i < n_loc; ++i) {
                std::cout << "| ";
                for (long j = 0; j < n_loc; ++j) {
                    std::cout << std::setw(7) << sub_mat[i * n_loc + j] << " ";
                }
                std::cout << "|\n";
            }
            std::cout << std::flush;
        }

        MPI_Barrier(MPI_COMM_WORLD);  // Wait for printing to finish
    }
} // Print matrix.sub_mat


// Uses Cblas Dgemm to multiply matrices
inline void multiply_matrix(long M, long N, long K,
                            const double* A, const double* B, double* C)
{
    cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                M, N, K,
                1.0,
                A, K,   // lda = leading dimension of A (cols)
                B, N,   // ldb = leading dimension of B (cols)
                0.0,
                C, N);  // ldc = leading dimension of C (cols)
} // Uses Cblas Dgemm to multiply matrices



// OPERATOR OVERLOADS

template <typename M>
CMatrix<M> operator*(CMatrix<M> &A, CMatrix<M> &B) {
    assert(A.world_size == B.world_size); // make sure world size is same

    CMatrix<M> C(A.N1, B.N2);  // Final multiplied mat

    CMatrix<M> B_blob(A.N2, B.n_loc); // Assembled B column from all processes

    CMatrix<M> C_block(A.n_loc, B.n_loc); // Temp block for C

    long int chunk = B.n_loc * B.n_loc; // B AllGather count

    for (int r = 0; r < A.world_size; ++r) { 
        { // start time scope
            
        CTimer t("Extract B");
        B.extract_block(r);  // uses openmp prllisation

        } // end time scope
        

        assert(B.sub_mat.size() == (size_t)chunk);
        assert(B_blob.data.size() == (size_t)(B_blob.N1 * B_blob.N2));

        { // start time scope
        CTimer t("MPI AllGather");

        MPI_Allgather(
            B.sub_mat.data(), 
            chunk, MPI_DOUBLE,
            B_blob.data.data(),
            chunk, MPI_DOUBLE,
            MPI_COMM_WORLD
        );
        }   // end time scope

        {// start time scope
        CTimer t("DGEMM");

        multiply_matrix(
            A.n_loc, B.n_loc, A.N2,
            A.data.data(),         // A: n_loc × N
            B_blob.data.data(),    // B: N × n_loc
            C_block.data.data()    // C_block: n_loc × n_loc
        );
        } // end time scope


        { // start time scope
        CTimer t("Write to C");
        
        long col_offset = r * B.n_loc;
        // #pragma omp parallel for collapse(2) schedule(static)
        for (long i = 0; i < A.n_loc; ++i) {
            for (long j = 0; j < B.n_loc; ++j) {
                C.data[i * C.N2 + (col_offset + j)] = C_block.data[i * B.n_loc + j];
                }
            }
        } // end time scope
    }

    return C;
}
