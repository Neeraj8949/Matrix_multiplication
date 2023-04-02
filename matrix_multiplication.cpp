#include <hpx/hpx_main.hpp>
#include <hpx/include/iostreams.hpp>
#include <hpx/include/lcos.hpp>
#include <hpx/include/parallel_for_each.hpp>
#include <hpx/include/parallel_generate.hpp>
#include <hpx/include/parallel_transform_reduce.hpp>

#include <vector>
#include <chrono>

#define MATRIX_SIZE 1000

using namespace std::chrono;

double matrix_a[MATRIX_SIZE][MATRIX_SIZE], matrix_b[MATRIX_SIZE][MATRIX_SIZE], matrix_c[MATRIX_SIZE][MATRIX_SIZE];

// Function to initialize the matrices with random values
void initialize_matrices()
{
    std::vector<double> temp(MATRIX_SIZE * MATRIX_SIZE);
    hpx::parallel::generate(hpx::parallel::execution::par, temp.begin(), temp.end(), []() { return (double)rand() / RAND_MAX; });

    for (int i = 0; i < MATRIX_SIZE; i++)
    {
        for (int j = 0; j < MATRIX_SIZE; j++)
        {
            matrix_a[i][j] = temp[i * MATRIX_SIZE + j];
            matrix_b[i][j] = temp[i * MATRIX_SIZE + j];
            matrix_c[i][j] = 0.0;
        }
    }
}

// Function to perform matrix multiplication
void matrix_multiply()
{
    high_resolution_clock::time_point t1 = high_resolution_clock::now();

    hpx::parallel::for_each(hpx::parallel::execution::par, hpx::make_counting_iterator(0), hpx::make_counting_iterator(MATRIX_SIZE),
                            [&](int i) {
                                for (int j = 0; j < MATRIX_SIZE; j++)
                                {
                                    for (int k = 0; k < MATRIX_SIZE; k++)
                                    {
                                        matrix_c[i][j] += matrix_a[i][k] * matrix_b[k][j];
                                    }
                                }
                            });

    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<milliseconds>(t2 - t1).count();

    hpx::cout << "Matrix multiplication completed in " << duration << " milliseconds." << hpx::endl;
}

int main()
{
    initialize_matrices();
    matrix_multiply();

    return 0;
}
