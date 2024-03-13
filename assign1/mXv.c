#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <mpi.h>
#include <time.h>

#define MAX_SIZE 32768
#define TILE_SIZE 16

// Function to allocate memory for matrix and vector
double** allocate_matrix(int rows, int cols) {
    double** matrix = (double**)malloc(rows * sizeof(double*));
    if (matrix == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    for (int i = 0; i < rows; i++) {
        matrix[i] = (double*)malloc(cols * sizeof(double));
        if (matrix[i] == NULL) {
            printf("Memory allocation failed!\n");
            return NULL;
        }
    }
    return matrix;
}

double* allocate_vector(int size) {
    double* vector = (double*)malloc(size * sizeof(double));
    if (vector == NULL) {
        printf("Memory allocation failed!\n");
        return NULL;
    }
    return vector;
}

// Function to fill matrix and vector with random values
void fill_random(double** matrix, double* vector, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            matrix[i][j] = (double)rand() / RAND_MAX;
        }
    }
    for (int i = 0; i < cols; i++) {
        vector[i] = (double)rand() / RAND_MAX;
    }
}

// Sequential matrix-vector multiplication
void sequential_mvm(double** matrix, double* vector, double* result, int rows, int cols) {
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// OpenMP naive matrix-vector multiplication
void openmp_mvm(double** matrix, double* vector, double* result, int rows, int cols) {
#pragma omp parallel for
    for (int i = 0; i < rows; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
}

// MPI naive matrix-vector multiplication
void mpi_mvm(double** matrix, double* vector, double* result, int rows, int cols, int rank, int size) {
    int start = (rows / size) * rank;
    int end = (rank == size - 1) ? rows : (rows / size) * (rank + 1);
    for (int i = start; i < end; i++) {
        result[i] = 0.0;
        for (int j = 0; j < cols; j++) {
            result[i] += matrix[i][j] * vector[j];
        }
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, result, rows / size, MPI_DOUBLE, MPI_COMM_WORLD);
}

// OpenMP matrix-vector multiplication with tiling
void openmp_tiled_mvm(double** matrix, double* vector, double* result, int rows, int cols, int tile_size) {
#pragma omp parallel for collapse(2)
    for (int i = 0; i < rows; i += tile_size) {
        for (int j = 0; j < cols; j += tile_size) {
            for (int ii = i; ii < i + tile_size; ii++) {
                for (int jj = j; jj < j + tile_size; jj++) {
                    result[ii] += matrix[ii][jj] * vector[jj];
                }
            }
        }
    }
}

// MPI matrix-vector multiplication with tiling
void mpi_tiled_mvm(double** matrix, double* vector, double* result, int rows, int cols, int rank, int size, int tile_size) {
    int start = (rows / size) * rank;
    int end = (rank == size - 1) ? rows : (rows / size) * (rank + 1);
    for (int i = start; i < end; i += tile_size) {
        for (int j = 0; j < cols; j += tile_size) {
            for (int ii = i; ii < i + tile_size; ii++) {
                for (int jj = j; jj < j + tile_size; jj++) {
                    result[ii] += matrix[ii][jj] * vector[jj];
                }
            }
        }
    }
    MPI_Allgather(MPI_IN_PLACE, 0, MPI_DATATYPE_NULL, result, rows / size, MPI_DOUBLE, MPI_COMM_WORLD);
}

// Function to calculate time
double get_time() {
#ifdef _OPENMP
    return omp_get_wtime();
#else
    return MPI_Wtime();
#endif
}

// Function to run and benchmark different programs
void run_programs(int program_type, int N, int rank, int size, FILE* fp) {
    double** matrix = allocate_matrix(N, N);
    if (matrix == NULL) {
        return;
    }
    double* vector = allocate_vector(N);
    if (vector == NULL) {
        free(matrix);
        return;
    }
    double* result = allocate_vector(N);
    if (result == NULL) {
        free(matrix);
        free(vector);
        return;
    }
    fill_random(matrix, vector, N, N);

    double start_time, end_time;
    double total_time = 0.0;

    switch (program_type) {
    case 1: // Sequential
        start_time = get_time();
        sequential_mvm(matrix, vector, result, N, N);
        end_time = get_time();
        total_time = end_time - start_time;
        fprintf(fp, "Sequential, %d, %.6f\n", N, total_time);
        break;
    case 2: // OpenMP naive
        start_time = get_time();
        openmp_mvm(matrix, vector, result, N, N);
        end_time = get_time();
        total_time = end_time - start_time;
        fprintf(fp, "OpenMP, %d, %.6f\n", N, total_time);
        break;
    case 3: // MPI naive
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = get_time();
        mpi_mvm(matrix, vector, result, N, N, rank, size);
        MPI_Barrier(MPI_COMM_WORLD);
        end_time = get_time();
        total_time = end_time - start_time;
        fprintf(fp, "MPI, %d, %.6f\n", N, total_time);
        break;
    case 4: // OpenMP tiled
        start_time = get_time();
        openmp_tiled_mvm(matrix, vector, result, N, N, TILE_SIZE);
        end_time = get_time();
        total_time = end_time - start_time;
        fprintf(fp, "OpenMP Tiled, %d, %.6f\n", N, total_time);
        break;
    case 5: // MPI tiled
        MPI_Barrier(MPI_COMM_WORLD);
        start_time = get_time();
        mpi_tiled_mvm(matrix, vector, result, N, N, rank, size, TILE_SIZE);
        MPI_Barrier(MPI_COMM_WORLD);
        end_time = get_time();
        total_time = end_time - start_time;
        fprintf(fp, "MPI Tiled, %d, %.6f\n", N, total_time);
        break;
    default:
        break;
    }

    free(matrix);
    free(vector);
    free(result);
}

int main(int argc, char* argv[]) {
    int rank, size;
    MPI_Init(&argc, &argv);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    srand((unsigned int)time(NULL));

    printf("Choose a program to run:\n");
    printf("1. Sequential\n");
    printf("2. OpenMP Naive\n");
    printf("3. MPI Naive\n");
    printf("4. OpenMP Tiled\n");
    printf("5. MPI Tiled\n");

    int choice;
    while (scanf_s("%d", &choice) != 1 || choice < 1 || choice > 5) {
        printf("Invalid choice! Please enter a number between 1 and 5: ");
        // Clear input buffer
        while (getchar() != '\n');
    }

    FILE* fp;
    if (fopen_s(&fp, "results.csv", "w") != 0) {
        printf("Error opening file!\n");
        MPI_Finalize();
        return 1;
    }

    fprintf(fp, "Test S.no, File, Input size, Time taken, Average so far\n");

    for (int N = 64; N <= MAX_SIZE; N *= 2) {
        run_programs(choice, N, rank, size, fp);
    }

    fclose(fp);

    MPI_Finalize();
    return 0;
}
