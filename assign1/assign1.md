# Road to matrix matrix multiplication.
#### NOTE: All code must be extensively commented, well formatted and indented. Be consistent and use either camelCase or snake_case. 
## Aim:
- The aim of this assignment is to learn parallel algorithm design on Omp and MPI
- Benchmark algorithms
## Learning Objective
After completion of the assignment you must be able to: -
- read and write OpenMP and MPI code for shared-memory and distributed systems.
- Code and use a helper file in batch or shell to benchmark programs. 
# Task:
### Overview
In this assignment you have to convert the class activity of matrix-vector multiplication into running code and benchmark it using a batch script. To do this first you need to make a sequential C program for matrix-vector multiplication, then a naive matrix-vector multiplication program using OpenMP and MPI both. Then make an advance matrix-vector multiplication program that uses tiling as discussed in class. Finally write a batch program that runs these 5  programs for different sizes of matrices and vectors. 
### Details
1. All the programs must make use of: -
     - dynamic memory allocation, 
     - command line arguments to take input for matrix size and vector size,
     - rand() function to fills the matrix and vector with some values.  
2. Write a sequential matrix-vector multiplication program in C.
3. Write an OpenMP naive matrix-vector multiplication program in C.
4. WRite an MPI naive matrix-vector multiplication program in C.
5. Write an OpenMP matrix-vector multiplication program in C that uses tiling of matrix.
6. Write an MPI matrix-vector multiplication program in C that uses tiling of matrix.
8. Write a windows batch or Linux shell script that runs these 5 programs and times each of them, for various sizes of inputs starting from 64, 128, 256, 512, 1024, and so on till 32768. You may stop at 16384 if your system is not able to compute the result in reasonable time (approx 20 min to 30 min is reasonable time).
9. The script in 8, must run each program for each input at least 10 times to get a good average time. see the sample input below. 
````
test S.no, file, input size, time taken, average so far 
01, mXv.exe, 64, 0.0056 secs, 0.0056 secs
02, mXv.exe, 64, 0.0049 secs, 0.00525 secs
03, mXv.exe, 64, 0.0051 secs, 0.0052 secs
:
: 
10, mXv.exe, 64, 0.0061 secs, 0.005425 secs
01, mXv_openmp.exe, 64, 0.0034 secs, 0.0034 secs
02, mXv_openmp.exe, 64, 0.0035 secs, 0.00345 secs
03, mXv_openmp.exe, 64, 0.0031 secs, 0.00333333 secs
:
:
````
10. Use the output like the sample above to import to excel or any spreadsheet program in CSV format. You may omit calculating average in the script file and get average in the spreadsheet program. 
11. Show the results and give your analysis in the readme.md file of the assignment folder in the repository. Attach screenshots of the results. Add all code files. Show bar-graphs, line-graphs etc where appropriate.
