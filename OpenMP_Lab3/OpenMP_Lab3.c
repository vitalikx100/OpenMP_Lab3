#include <omp.h>
#include <stdio.h>
#include <math.h>
#include <stdlib.h>

#define NMAX 3200000
#define Q 28
#define CHUNK 1000

int main(int argc, char* argv[])
{
    printf("Variant 19: double, 7, 3 200 000, [4, 8, 16], 28\n");

    omp_set_num_threads(16);

    double* mass_1 = (double*)malloc(NMAX * sizeof(double));
    double* mass_2 = (double*)malloc(NMAX * sizeof(double));
    double* mass_3 = (double*)malloc(NMAX * sizeof(double));
    double* mass_4 = (double*)malloc(NMAX * sizeof(double));
    double* mass_5 = (double*)malloc(NMAX * sizeof(double));
    double* mass_6 = (double*)malloc(NMAX * sizeof(double));
    double* mass_7 = (double*)malloc(NMAX * sizeof(double));

    double* sum_sequence = (double*)calloc(NMAX, sizeof(double));
    double* sum_static = (double*)calloc(NMAX, sizeof(double));
    double* sum_dynamic = (double*)calloc(NMAX, sizeof(double));
    double* sum_guided = (double*)calloc(NMAX, sizeof(double));

    double t_s = 0, t_p = 0, t_st = 0, t_d = 0, t_g = 0;
    double start_time;
    const int reps = 20;
    int i, rep, q;

    for (int i = 0; i < NMAX; i++) {
        mass_1[i] = 1.0;
        mass_2[i] = 1.0;
        mass_3[i] = 1.0;
        mass_4[i] = 1.0;
        mass_5[i] = 1.0;
        mass_6[i] = 1.0;
        mass_7[i] = 1.0;
    }

    for (rep = 0; rep < reps; rep++)
    {
        start_time = omp_get_wtime();
#pragma omp parallel
        {

        }
        t_p += omp_get_wtime() - start_time;

        start_time = omp_get_wtime();
        for (i = 0; i < NMAX; i++) {
            for (q = 0; q < Q; q++) {
                sum_sequence[i] += mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
            }
        }
        t_s += omp_get_wtime() - start_time;

#pragma omp parallel
        {
            start_time = omp_get_wtime();
#pragma omp for schedule(static, CHUNK) private(i, q)
            for (i = 0; i < NMAX; i++) {
                for (q = 0; q < Q; q++) {
                    sum_static[i] += mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
                }
            }
            t_st += omp_get_wtime() - start_time;


            start_time = omp_get_wtime();
#pragma omp for schedule(dynamic, CHUNK) private(i, q)
            for (i = 0; i < NMAX; i++) {
                for (q = 0; q < Q; q++) {
                    sum_dynamic[i] += mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
                }
            }
            t_d += omp_get_wtime() - start_time;


            start_time = omp_get_wtime();
#pragma omp for schedule(guided, CHUNK) private(i, q)
            for (i = 0; i < NMAX; i++) {
                for (q = 0; q < Q; q++) {
                    sum_guided[i] += mass_1[i] + mass_2[i] + mass_3[i] + mass_4[i] + mass_5[i] + mass_6[i] + mass_7[i];
                }
            }
            t_g += omp_get_wtime() - start_time;
        }
    }

    t_p /= reps; t_s /= reps; t_st /= reps; t_d /= reps; t_g /= reps;

    for (i = 0; i < NMAX; i++) {
        sum_sequence[i] /= reps * Q;
        sum_static[i] /= reps * Q;
        sum_dynamic[i] /= reps * Q;
        sum_guided[i] /= reps * Q;
    }

    printf("Sum sequence (0, 1, ... , NMAX-2, NMAX-1): %f %f ... %f %f\n", sum_sequence[0], sum_sequence[1], sum_sequence[NMAX - 2], sum_sequence[NMAX - 2]);
    printf("Sum static (0, 1, ... , NMAX-2, NMAX-1): %f %f ... %f %f\n", sum_static[0], sum_static[1], sum_static[NMAX - 2], sum_static[NMAX - 2]);
    printf("Sum dynamic (0, 1, ... , NMAX-2, NMAX-1): %f %f ... %f %f\n", sum_dynamic[0], sum_dynamic[1], sum_dynamic[NMAX - 2], sum_dynamic[NMAX - 2]);
    printf("Sum guided (0, 1, ... , NMAX-2, NMAX-1): %f %f ... %f %f\n", sum_guided[0], sum_guided[1], sum_guided[NMAX - 2], sum_guided[NMAX - 2]);




    printf("\nAverage time:\n");
    printf("Initialization parallel area: %f seconds\n", t_p);
    printf("Sequence algorithm time: %f seconds.\n", t_s);
    printf("For schedule STATIC time: %f seconds.\n", t_st);
    printf("For schedule DYNAMIC time: %f seconds.\n", t_d);
    printf("For schedule GUIDED time: %f seconds.\n", t_g);

    double a_st = t_s / t_st; double a_d = t_s / t_d; double a_g = t_s / t_g;
    double a_stp = t_s / (t_st + t_p); double a_dp = t_s / (t_d + t_p); double a_gp = t_s / (t_g + t_p);

    printf("\nCalculating acceleration:\n");
    printf("1. Without parallel initialization:\n");
    printf("Static a_st: %f \n", a_st);
    printf("Dynamic a_d: %f \n", a_d);
    printf("Guided a_g: %f \n\n", a_g);

    printf("2. With parallel initialization:\n");
    printf("Static a_stp: %f \n", a_stp);
    printf("Dynamic a_dp: %f \n", a_dp);
    printf("Guided a_gp: %f \n", a_gp);


    free(mass_1);
    free(mass_2);
    free(mass_3);
    free(mass_4);
    free(mass_5);
    free(mass_6);
    free(mass_7);

    return 0;



}