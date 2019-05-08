// Copyright (c) Microsoft Corporation. All rights reserved.
// Licensed under the MIT license.

#include <cassert>
#include <chrono>
#include "bof_utils.h"
#include "flash_blas.h"
#include "lib_funcs.h"

using namespace std::chrono;

std::string mnt_dir = "/tmp/gemm_driver_temps";

flash::Logger logger("gemm_driver");

int main(int argc, char** argv) {
  if (argc != 15) {
    LOG_INFO(logger,
             "Usage Mode : <exec> <mat_A_file> <mat_B_file> <mat_C_file> "
             "<A_nrows> <A_ncols> <B_ncols> <alpha> <beta> <a transpose?> <b "
             "transpose?> <matr order> <lda_a> <lda_b> <lda_c>");
    LOG_FATAL(logger, "expected 14 args, got ", argc - 1);
  }

  // init blas-on-flash
  LOG_DEBUG(logger, "setting up flash context");
  flash::flash_setup(mnt_dir);

  // map matrices to flash pointers
  std::string A_name = std::string(argv[1]);
  std::string B_name = std::string(argv[2]);
  std::string C_name = std::string(argv[3]);
  
  float* mat_A = (float*) malloc(m * k * sizeof(float));
  float* mat_B = (float*) malloc(n * k * sizeof(float));
  float* mat_C = (float*) malloc(m * n * sizeof(float));

  LOG_INFO(logger, "Reading matrix A into memory");
  std::ifstream a_file(A_name, std::ios::binary);
  a_file.read((char*) mat_A, m * k * sizeof(float));
  a_file.close();
  LOG_INFO(logger, "Reading matrix B into memory");
  std::ifstream b_file(B_name, std::ios::binary);
  b_file.read((char*) mat_B, k * n * sizeof(float));
  b_file.close();
  LOG_INFO(logger, "Reading matrix C into memory");
  std::ifstream c_file(C_name, std::ios::binary);
  c_file.read((char*) mat_C, m * n * sizeof(float));
  c_file.close();

  // problem dimension
  FBLAS_UINT m = (FBLAS_UINT) std::stol(argv[4]);
  FBLAS_UINT k = (FBLAS_UINT) std::stol(argv[5]);
  FBLAS_UINT n = (FBLAS_UINT) std::stol(argv[6]);
  FPTYPE     alpha = (FPTYPE) std::stof(argv[7]);
  FPTYPE     beta = (FPTYPE) std::stof(argv[8]);
  CHAR       trans_a = argv[9][0];
  CHAR       trans_b = argv[10][0];
  CHAR       mat_ord = argv[11][0];
  FBLAS_UINT lda_a = (FBLAS_UINT) std::stol(argv[12]);
  FBLAS_UINT lda_b = (FBLAS_UINT) std::stol(argv[13]);
  FBLAS_UINT lda_c = (FBLAS_UINT) std::stol(argv[14]);

  LOG_INFO(logger, "dimensions : A = ", m, "x", k, ", B = ", k, "x", n);

  // execute gemm call
  high_resolution_clock::time_point t1 = high_resolution_clock::now();
  FBLAS_INT res = flash::gemm(mat_ord, trans_a, trans_b, m, n, k, alpha, beta,
                              mat_A, mat_B, mat_C, lda_a, lda_b, lda_c);
  high_resolution_clock::time_point t2 = high_resolution_clock::now();
  duration<double> span = duration_cast<duration<double>>(t2 - t1);
  LOG_INFO(logger, "gemm() took ", span.count());

  LOG_INFO(logger, "flash::gemm() returned with ", res);

  LOG_DEBUG(logger, "un-map matrices");
  // unmap files
  LOG_INFO(logger, "Writing C to file");
  std::ofstream cout_file(C_name, std::ios::binary);
  cout_file.write((char*) mat_C, m * n * sizeof(float));
  cout_file.close();

  LOG_DEBUG(logger, "destroying flash context");
  // destroy blas-on-flash
}
