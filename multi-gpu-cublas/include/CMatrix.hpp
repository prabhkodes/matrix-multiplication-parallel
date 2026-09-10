#pragma once
#include <algorithm> 
#include <iostream>
#include <vector>
#include <cstdlib>   
#include <ctime>     
#include <cblas.h> 
#include <mpi.h>
#include <iomanip> 
#include <cassert> 

#include <cublas_v2.h>

#include "../include/parallel_2.hpp"




template <typename T>
class CMatrix{
public:
    std::vector<T> data; //Actual matrix data
    std::vector<T> sub_mat; // sub matrix 

    long int N1, N2; // N1 -> no. of rows, N2 -> no. of columns
    long int n_loc;

    size_t size_in_bytes; // used for cudaMalloc

    T* dev_data = nullptr; // device local data

    int world_rank, world_size; // MPI world rank and size upon init of CMatrix
    
    CMatrix(long int N1, long int N2); // Ctor
    ~CMatrix(); // Dctor
    
    void fill_rand(); // fills random T values
    void fill_identity(); // Makes diagnol elements 1 

    void print_mat_with_label(const std::string& label) const;
    void print_submat_with_label(const std::string& label) const;

    void extract_block(int iter);

    template<typename M>
    friend CMatrix<M> operator* (const CMatrix<M>& m1, const CMatrix<M>& m2);
};


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

    size_in_bytes = N1 * N2 * sizeof(T);

    // cudaMalloc device data
    cudaMalloc((void**)&dev_data, size_in_bytes);

} // Constructor



template <typename T>
CMatrix<T>::~CMatrix() { // Desctructor
    if (dev_data != nullptr) {
        cudaFree(dev_data);
        dev_data = nullptr;
    }
} // Desctructor


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

    // Copy Host matrix to device
    cudaMemcpy(dev_data, data.data(), size_in_bytes, cudaMemcpyHostToDevice);

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

    // Copy Host matrix to device
    cudaMemcpy(dev_data, data.data(), size_in_bytes, cudaMemcpyHostToDevice);

} // Diag elems 1 and rest 0 for matrix.data


template <typename T>
void CMatrix<T>::extract_block(int iter) {
    const long int col_start_offset = iter * n_loc;

    assert((col_start_offset + n_loc) <= N2);  
   
    sub_mat.resize(n_loc * n_loc);  

    for (long int i = 0; i < n_loc; ++i) {
        for (long int j = 0; j < n_loc; ++j) {
            sub_mat[i * n_loc + j] = data[i * N2 + (col_start_offset + j)];
        }
    }
}



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



inline void multiply_matrix_cublas(long M, long N, long K,
                                   const double* dev_A, const double* dev_B, double* dev_C, cublas_handle)
{
    // Dimensions of the operation C^T = B^T * A^T
    long m_gemm = N; // Rows of C^T (Cols of C)
    long n_gemm = M; // Cols of C^T (Rows of C)
    long k_gemm = K; // Inner dimension
    
    const double alpha = 1.0;
    const double beta = 0.0;

    // nvtxRangePush("CuBlas DGEMM");
    {// start time scope
    CTimer t("DGEMM");
    cublasDgemm(
        cublas_handle,
        CUBLAS_OP_N,          // transa: No Transpose (Treat B_RowMaj as B^T_ColMaj)
        CUBLAS_OP_N,          // transb: No Transpose (Treat A_RowMaj as A^T_ColMaj)
        m_gemm,               // M: Rows of result C^T -> N
        n_gemm,               // N: Cols of result C^T -> M
        k_gemm,               // K: Inner dim -> K
        &alpha,
        dev_B,                // A Matrix in Gemm (First Operand is B)
        N,                    // lda: Leading dim of B_RowMaj (Columns of B) -> N
        dev_A,                // B Matrix in Gemm (Second Operand is A)
        K,                    // ldb: Leading dim of A_RowMaj (Columns of A) -> K
        &beta,
        dev_C,                // C Matrix
        N                     // ldc: Leading dim of C_RowMaj (Columns of C) -> N
    );
    cudaDeviceSynchronize();
    } // end timer scope
    
    // nvtxRangePop();
}



// OPERATOR OVERLOADS

template <typename M>
CMatrix<M> operator*(CMatrix<M> &A, CMatrix<M> &B) {
    assert(A.world_size == B.world_size); // make sure world size is same

    CMatrix<M> C(A.N1, B.N2);  // Final multiplied mat

    CMatrix<M> B_blob(A.N2, B.n_loc); // Assembled B column from all processes

    CMatrix<M> C_block(A.n_loc, B.n_loc); // Temp block for C

    long int B_blob_size_bytes = B_blob.size_in_bytes; 
    long int C_block_size_bytes = C_block.size_in_bytes;

    long int chunk = B.n_loc * B.n_loc; // B AllGather count

    cublasHandle_t cublas_handle;
    cublasCreate(&cublas_handle);

    for (int r = 0; r < A.world_size; ++r) {         
        { // start time scope
        CTimer t("Extract B");
        B.extract_block(r);  
        } // end time scope


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

        // Copy B_blob from host to device after communication
        cudaMemcpy(B_blob.dev_data, B_blob.data.data(), B_blob_size_bytes, cudaMemcpyHostToDevice);


        multiply_matrix_cublas( 
            A.n_loc, B.n_loc, A.N2,
            A.dev_data,            // A: USE DEVICE POINTER
            B_blob.dev_data,       // B: USE DEVICE POINTER
            C_block.dev_data ,      // C_block: USE DEVICE POINTER
            cublas_handle     
        );


        // Write C_block from device to host
        cudaMemcpy(C_block.data.data(), C_block.dev_data, C_block_size_bytes, cudaMemcpyDeviceToHost);

        { // start time scope
        CTimer t("Write to C");

        // Copy C_block into the correct columns of C
        long col_offset = r * B.n_loc;
        for (long i = 0; i < A.n_loc; ++i) {
            for (long j = 0; j < B.n_loc; ++j) {
                C.data[i * C.N2 + (col_offset + j)] = C_block.data[i * B.n_loc + j];
                }
            }
        } // end time scope
    }
    cublasDestroy(cublas_handle);
    return C;
}
