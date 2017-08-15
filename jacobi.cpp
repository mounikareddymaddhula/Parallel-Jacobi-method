/**
 * @file    jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements matrix vector multiplication and Jacobi's method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */
#include "jacobi.h"

/*
 * TODO: Implement your solutions here
 */

// my implementation:
#include <iostream>
#include <math.h>
#include <stdlib.h>
#include <stdio.h>

double l2_norm(const int n, double* A, double* b, double* x)
{
  double* temp = (double*)malloc(n*sizeof(double));
  matrix_vector_mult(n, A, x, temp);
  for(int i = 0; i < n; i++) temp[i] -= b[i];
   double sum = 0; 
   for (int i = 0; i < n ; i++)
     sum += temp[i]*temp[i];
   return sqrt(sum);
}




// Calculates y = A*x for a square n-by-n matrix A, and n-dimensional vectors x
// and y
void matrix_vector_mult(const int n, const double* A, const double* x, double* y)
{
   for (int i = 0; i < n; i++)
   {
      y[i] = 0;
      for (int j = 0; j < n; j++)
      {
           y[i] += (*(A + i*n + j) * x[j]);
      }
   }
}

// Calculates y = A*x for a n-by-m matrix A, a m-dimensional vector x
// and a n-dimensional vector y
void matrix_vector_mult(const int n, const int m, const double* A, const double* x, double* y)
{
    for (int i = 0; i <n; i++)
    {
       y[i] = 0;
       for (int j = 0; j < m; j++)
       {
          y[i] += (*(A + i*n + j) * x[j]);
       }
    }
}

// implements the sequential jacobi method
void jacobi(const int n, double* A, double* b, double* x, int max_iter, double l2_termination)
{
    for (int i = 0; i < n; i++)
      x[i] = 0;
    double* D = (double*)malloc(n*n*sizeof(double));
    double* R = (double*)malloc(n*n*sizeof(double));
    double* D_inv = (double*)malloc(n*n*sizeof(double));
    for (int i = 0; i < n; i++)
    {
       for (int j = 0; j < n; j++)
       {
         if (i == j) 
         {
             *(D + i*n + j) = *(A + i*n +j);
             *(R + i*n + j) = 0;
         }
         else
         {
             *(D + i*n + j) = 0;
             *(R + i*n + j) = *(A + i*n + j);
         }
       }
    }
    for (int i = 0; i < n ; i++)
    {
       for (int j =0; j <n; j++)
       {
          if (i == j) 
            *(D_inv + i*n + j) = 1/(*(D + i*n + j));
          else
             *(D_inv + i*n + j) = 0;
       }
    }   
   int count = 0;
   while (count < max_iter && l2_norm(n, A, b, x) > l2_termination)
   {
     double* temp = (double*)malloc(n*sizeof(double));
     matrix_vector_mult(n, R, x, temp);
     for (int i = 0; i <n; i++) temp[i] = b[i] - temp[i];
     matrix_vector_mult(n, D_inv, temp, x);
     count++;
   }
   free(D);
   free(R);
   free(D_inv);
}






