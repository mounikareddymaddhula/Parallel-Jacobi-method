/**
 * @file    mpi_jacobi.cpp
 * @author  Patrick Flick <patrick.flick@gmail.com>
 * @brief   Implements MPI functions for distributing vectors and matrixes,
 *          parallel distributed matrix-vector multiplication and Jacobi's
 *          method.
 *
 * Copyright (c) 2014 Georgia Institute of Technology. All Rights Reserved.
 */

#include "mpi_jacobi.h"
#include "jacobi.h"
#include "utils.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <vector>

/*
 * TODO: Implement your solutions here
 */

double l2_norm2(const int n, double* A, double* b, double* x)
{
  double* temp = (double*)malloc(n*sizeof(double));
  matrix_vector_mult(n, A, x, temp);
  for(int i = 0; i < n; i++) temp[i] -= b[i];
   double sum = 0; 
   for (int i = 0; i < n ; i++)
     sum += temp[i]*temp[i];
   return sqrt(sum);
}



void distribute_vector(const int n, double* input_vector, double** local_vector, MPI_Comm comm)
{
    int rank;
    //int coords[2] = {0, 0};
    //MPI_Cart_rank(grid_comm, coords, &rank);
    MPI_Comm_rank(comm, &rank);
    int p,q;
    MPI_Comm_size(comm,&p);
    q = sqrt(p);
    int coords[2];
    MPI_Cart_coords(comm,rank,2,coords);
    int src, dst;
    if (rank == 0)
    {
#if 0
      for(int j = 0; j < n; j++)
        printf("Before scatter: Rank is : %d and value is %f\n", rank, *(input_vector+j));
#endif
      int sent =0;
      for (int i = 0; i < q; i++)
      {
         MPI_Cart_shift(comm, 0, i, &src, &dst);
         int tosend = 0;
         int coordst[2];
         MPI_Cart_coords(comm,dst,2,coordst);
         int cei,flr;
	 if(n%q >0){
		cei =(int)(n/q)+1;
		flr = (int)(n/q);
	}
	else{
   		cei =(int)(n/q);
		flr = (int)(n/q);
	} 
        
        if( coordst[0] < (n%q))
           tosend = cei;
         else
           tosend = flr;

#if 0  
         if( coordst[0] < (n%q))
           tosend = ceil(n/q);
         else
           tosend = floor(n/q);
#endif
         MPI_Request request;
         MPI_Isend((&input_vector[sent]), tosend, MPI_DOUBLE, dst, 0, comm, &request);
#if 0
         for (int i = 0; i < tosend; i++)
         {
           printf("---Rank is: %d and coords is %d and value sent is: %lf \n", dst, coordst[1], input_vector[sent+i] );
         }
#endif 
         sent += tosend;
      }
    }
    int torcv;
    //MPI_Barrier(comm);
    if(coords[1] == 0)
    {
      //int torcv;
      int cei,flr;
      if((n%q) > 0)
      {
	 cei =(int)(n/q)+1;
	 flr = (int)(n/q);
      }
      else
      {
   	 cei =(int)(n/q);
	 flr = (int)(n/q);
      }   
      if( coords[0] < (n%q))
          torcv = cei;
      else
          torcv = flr;

      *local_vector = (double*)malloc(torcv*sizeof(double));
      for (int i=0; i < torcv; i++)
          ((*local_vector)[i]) = 10;   
 
      MPI_Status status;
      double* temp = (double*)malloc(torcv*sizeof(double));
      MPI_Recv(&((*local_vector)[0]), torcv, MPI_DOUBLE, 0, 0, comm, &status);
      //MPI_Recv(temp, torcv, MPI_DOUBLE, 0, 0, comm, &status);
      //printf("Rank is %d and torcv is %d \n", rank, torcv);
#if 0
      for (int i = 0; i < torcv; i++)
      {
        printf("Rank is: %d and value is: %lf \n", rank, ((*local_vector)[i]));
        //printf("Rank is: %d and value is: %lf \n", rank, temp[i]);
      }
#endif
    }
    //MPI_Barrier(comm);
}


// gather the local vector distributed among (i,0) to the processor (0,0)
void gather_vector(const int n, double* local_vector, double* output_vector, MPI_Comm comm)
{
    int rank;
    MPI_Comm_rank(comm, &rank);
    int p,q;
    MPI_Comm_size(comm,&p);
    q = sqrt(p);
    int coords[2];
    MPI_Cart_coords(comm,rank,2,coords);
    int src, dst;

    int tosend;
    int cei,flr;
      if((n%q) > 0)
      {
	 cei =(int)(n/q)+1;
	 flr = (int)(n/q);
      }
      else
      {
   	 cei =(int)(n/q);
	 flr = (int)(n/q);
      }   

    if (coords[1] == 0)
    {  
      if (coords[0] < n%q)
      {
        tosend = cei;
      }
      else
      {
        tosend = flr;
      }
    }
    else 
      tosend = 0;
#if 0
    for (int i = 0; i < tosend; i++)
    {
      printf("Rank is: %d and value is: %f\n", rank, *(local_vector + i));
    }
#endif 
    int *rcv_count = (int*)malloc(q*q*sizeof(int));
    int *rcv_displ = (int*)malloc(q*q*sizeof(int));
    int coords2[2];
    for (int i = 0; i < q; i++)
    {
      for(int j = 0; j < q; j++)
      {
        coords2[0]=i;
        coords2[1]=j;
        if(j != 0)
          *(rcv_count+i*q+j)=0;
        else
          if(i < n%q)
            *(rcv_count+i*q+j) = cei;
          else
            *(rcv_count+i*q+j) = flr;
      }
    }
    int sum = 0;
    for (int i = 0; i < q; i++)
    {
      for(int j = 0; j < q; j++)
      {
        *(rcv_displ+i*q+j) = sum;
        sum += *(rcv_count+i*q+j);
      }
    }
    MPI_Gatherv(local_vector, tosend, MPI_DOUBLE, output_vector, rcv_count, rcv_displ, MPI_DOUBLE, 0, comm);
    free(rcv_count);
    free(rcv_displ);

#if 0
    if(rank == 0)
      for(int j = 0; j < n; j++)
        printf("After gather: Rank is : %d and value is %f\n", rank, *(output_vector+j));
#endif
}

void distribute_matrix(const int n, double* input_matrix, double** local_matrix, MPI_Comm comm)
{
    int rank;
    //int coords[2] = {0, 0};
    //MPI_Cart_rank(grid_comm, coords, &rank);
    MPI_Comm_rank(comm, &rank);
    int p,q;
    MPI_Comm_size(comm,&p);
    q = sqrt(p);
    int coords[2];
    MPI_Cart_coords(comm,rank,2,coords);
    int src, dst;
    if(rank == 0)
    {
#if 0
      for(int j = 0; j < n*n; j++)
        printf("Before scatter: Rank is : %d and value is %f\n", rank, *(input_matrix+j));
#endif
      int sent =0;
      for(int i = 0; i<q; i++)
      {
        MPI_Cart_shift(comm, 0, i, &src, &dst);
	int coordst[2];
    	MPI_Cart_coords(comm,dst,2,coordst);
        int tosend = 0;
	int cei,flr;
	if(n%q >0){
		cei =(int)(n/q)+1;
		flr = (int)(n/q);
	}
	else{
		cei =(int)(n/q);
		flr = (int)(n/q);
	} 
	if(coordst[0] < (n%q))
          tosend = (cei*cei*(n%q)) + (cei*flr*(q-(n%q)));
        else
          tosend = (cei*flr*(n%q)) + (flr*flr*(q-(n%q)));
/*
        if(dst < (n%q))
          tosend = (ceil(n/q)*ceil(n/q)*(n%q)) + (ceil(n/q)*floor(n/q)*(q-(n%q)));
        else
          tosend = (ceil(n/q)*floor(n/q)*(n%q)) + (floor(n/q)*floor(n/q)*(q-(n%q)));
*/
        MPI_Request request;
        MPI_Isend((&input_matrix[sent]), tosend, MPI_DOUBLE, dst, 0, comm, &request);
#if 0     
        for (int i = 0; i < tosend; i++)
        {
           printf("---Rank is: %d and value sent is: %f \n", dst, input_matrix[sent+i] );
        }
#endif
        sent += tosend;
      }
    }
    if(coords[1] == 0)
    {
      int torcv;
      int cei,flr;
	if(n%q >0){
		cei =(int)(n/q)+1;
		flr = (int)(n/q);
	}
	else{
		cei =(int)(n/q);
		flr = (int)(n/q);
	} 
	if(coords[0] < (n%q))
          torcv = (cei*cei*(n%q)) + (cei*flr*(q-(n%q)));
        else
          torcv = (cei*flr*(n%q)) + (flr*flr*(q-(n%q)));
#if 0
      if(coords[0] < (n%q))
        torcv = ceil(n/q)*ceil(n/q)*(n%q) + ceil(n/q)*floor(n/q)*(q-(n%q));
      else
        torcv = ceil(n/q)*floor(n/q)*(n%q) + floor(n/q)*floor(n/q)*(q-(n%q));
#endif
      double* tmp_rcv = (double*) malloc(torcv*sizeof(double));
      MPI_Status status;
      MPI_Recv(tmp_rcv, torcv, MPI_DOUBLE, 0, 0, comm, &status);
#if 0
      for (int i = 0; i < torcv; i++)
      {
        printf("Rank is: %d and value is: %f \n", rank, *(tmp_rcv+i) );
      }
#endif 

      int sent = 0,i=0;
      while(sent < torcv)
      {
        MPI_Cart_shift(comm, 1, i, &src, &dst);
        int tosend=0;
        int coords2[2];
        MPI_Cart_coords(comm,dst,2,coords2);
#if 0
        if(coords2[0] < (n%q) && coords2[1] < (n%q))
          tosend = ceil(n/q);
        else if(coords2[0] < (n%q))
          tosend = floor(n/q);
        else if(coords2[0] > (n%q) && coords2[1] < (n%q))
          tosend = ceil(n/q)*floor(n/q);
        else if(coords2[0] > (n%q))
          tosend = floor(n/q)*floor(n/q);
#endif
        int cei,flr;
	if(n%q >0){
		cei =(int)(n/q)+1;
		flr = (int)(n/q);
	}
	else{
		cei =(int)(n/q);
		flr = (int)(n/q);
	}        

        if(coords2[1] < (n%q))
          tosend = cei;
        else
          tosend = flr;
        MPI_Request request;
        MPI_Isend((tmp_rcv + sent), tosend, MPI_DOUBLE, dst, 0, comm, &request);
        sent += tosend;
        i = (i+1)%q;
      }
      free(tmp_rcv);
    }
        int cei,flr;
	if(n%q >0){
		cei =(int)(n/q)+1;
		flr = (int)(n/q);
	}
	else{
		cei =(int)(n/q);
		flr = (int)(n/q);
	}

    int torcv=0;
    if( (coords[0] < (n%q)) && (coords[1] < (n%q)))
      torcv = cei*cei;
    else if(coords[0] < (n%q))
      torcv = cei*flr;
    else if((coords[0] >= (n%q)) && (coords[1] < (n%q)))
      torcv = cei*flr;
    else if(coords[0] >= (n%q))
      torcv = flr*flr;
    *local_matrix = (double*)malloc(torcv*sizeof(double));
    int recd = 0;
    int rc2;
    while(recd < torcv)
    {
      int cei,flr;
	if(n%q >0){
		cei =(int)(n/q)+1;
		flr = (int)(n/q);
	}
	else{
		cei =(int)(n/q);
		flr = (int)(n/q);
	}

      if(coords[1] < (n%q))
        rc2 = cei;
      else
        rc2 = flr;
      MPI_Status status;
      MPI_Recv((*local_matrix)+recd, rc2, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status);
      recd += rc2;
    }
#if 0
    for (int i = 0; i < torcv; i++)
    {
      printf("Rank is: %d, torcv is: %d and value is: %f\n", rank, torcv, *(*local_matrix + i));
    }
#endif
    MPI_Barrier(comm);
}


void transpose_bcast_vector(const int n, double* col_vector, double* row_vector, MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  int p,q;
  MPI_Comm_size(comm,&p);
  q = sqrt(p);
  int tosend;
  int coords[2];
  int src, dst;
  MPI_Cart_coords(comm,rank,2,coords);
  int cei,flr;
  if((n%q) > 0)
  {
     cei =(int)(n/q)+1;
     flr = (int)(n/q);
  }
  else
  {
     cei =(int)(n/q);
     flr = (int)(n/q);
  }
  if (coords[1] == 0)
  {
      MPI_Cart_shift(comm, 1, coords[0], &src, &dst);
      if(coords[0] < (n%q))
         tosend = cei;
      else
         tosend = flr;
      MPI_Request request;
      MPI_Isend(col_vector, tosend, MPI_DOUBLE, dst, 0, comm, &request);
  }  
  int torcv;

  if (coords[0] == coords[1])
  {
    if(coords[0] < (n%q))
        torcv = cei;
    else
        torcv = flr;
    MPI_Status status;
    MPI_Recv(row_vector, torcv, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status);
  }
  int color = coords[1];
  MPI_Comm col_comm;
  MPI_Comm_split(comm, color, coords[0], &col_comm);
  int col_rank, col_size;
  MPI_Comm_rank(col_comm, &col_rank);
  MPI_Comm_size(col_comm,&col_size);
  int count;
  if (color < n%q)
    count = cei;
  else
    count = flr;
  MPI_Bcast(row_vector, count, MPI_DOUBLE, coords[1], col_comm);
#if 0
  for(int i=0; i< count; i++)
    printf("Rank is %d, number is %f\n", rank, row_vector[i]);
#endif
}


void distributed_matrix_vector_mult(const int n, double* local_A, double* local_x, double* local_y, MPI_Comm comm)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  int p,q;
  MPI_Comm_size(comm,&p);
  q = sqrt(p);
  int tosend;
  int coords[2];
  int src, dst;
  MPI_Cart_coords(comm,rank,2,coords);
  int count;
  int cei,flr;
  if((n%q) > 0)
  {
     cei =(int)(n/q)+1;
     flr = (int)(n/q);
  }
  else
  {
     cei =(int)(n/q);
     flr = (int)(n/q);
  }
  if (coords[1] < n%q)
    count = cei;
  else
    count = flr;    
  double* row_vector_x = (double*)malloc(count*sizeof(double)); 
  transpose_bcast_vector(n, local_x, row_vector_x, comm);

  int rows;
  if(coords[0] < n%q)
    rows = cei;
  else
    rows = flr;
  double* local_y1 = (double*)malloc(rows*sizeof(double));
  for (int i = 0; i <rows; i++)
  {
    local_y1[i] = 0;
    for (int j = 0; j < count; j++)
    {
       local_y1[i] += (*(local_A + i*count + j) * row_vector_x[j]);
    }
  }

  int color = coords[0];
  MPI_Comm row_comm;
  MPI_Comm_split(comm, color, coords[1], &row_comm);
  int row_rank, row_size;
  MPI_Comm_rank(row_comm, &row_rank);
  MPI_Comm_size(row_comm,&row_size);
  MPI_Reduce(local_y1, local_y, rows, MPI_DOUBLE, MPI_SUM, 0, row_comm);
#if 0
  if (coords[1]==0)
  for(int i=0; i< rows; i++)
    printf("/////////Rank is %d, number is %f\n", rank, local_y[i]);
#endif

  free(local_y1);
  free(row_vector_x);
}

// Solves Ax = b using the iterative jacobi method
void distributed_jacobi(const int n, double* local_A, double* local_b, double* local_x,
                MPI_Comm comm, int max_iter, double l2_termination)
{
  int rank;
  MPI_Comm_rank(comm, &rank);
  int p,q;
  MPI_Comm_size(comm,&p);
  q = sqrt(p);
  int tosend;
  int coords[2];
  int src, dst;
  MPI_Cart_coords(comm,rank,2,coords);
  int cei,flr;
  if((n%q) > 0)
  {
    cei =(int)(n/q)+1;
    flr = (int)(n/q);
   }
   else
   {
     cei =(int)(n/q);
     flr = (int)(n/q);
   }

  int count=0;
  if( (coords[0] < (n%q)) && (coords[1] < (n%q)))
    count = cei*cei;
  else if(coords[0] < (n%q))
    count = cei*flr;
  else if((coords[0] >= (n%q)) && (coords[1] < (n%q)))
    count = cei*flr;
  else if(coords[0] >= (n%q))
    count = flr*flr;
  double *R = (double*)malloc(count*sizeof(double));

  int count1,count2;
  if(coords[0] < (n%q))
    count1 = cei;
  else
    count1 = flr;
  if(coords[1] < (n%q))
    count2 = cei;
  else
    count2 = flr;
  double *col_vector = (double*)malloc(count1*sizeof(double));
  if (coords[0] == coords[1])
  {
    for(int i=0; i<count1;i++)
    {
      for(int j=0; j<count2;j++)
      {
        if(i==j)
          {
            *(R + i*count2 + j) = 0;
            col_vector[i] = *(local_A + i*count2 + j);
          }
        else
          {
          *(R + i*count2 + j) = *(local_A + i*count2 + j);
          }
      }
    }
  }
  else
  {
    for(int i=0; i<count1;i++)
    {
      for(int j=0; j<count2;j++)
      {
        *(R + i*count2 + j) = *(local_A + i*count2 + j);
      }
    }
  }
  //int tosend;
  if (coords[0] == coords[1])
  {
    MPI_Cart_shift(comm, 1, -coords[1], &src, &dst);
    if(coords[0] < (n%q))
        tosend = cei;
    else
        tosend = flr;
    MPI_Request request;
    MPI_Isend(col_vector, tosend, MPI_DOUBLE, dst, 0, comm, &request);  
  }
  int torcv;
  double *D = (double*)malloc(count1*sizeof(double));
  if (coords[1] == 0)
  {    
      if(coords[0] < (n%q))
         torcv = cei;
      else
         torcv = flr;
      MPI_Status status;
      MPI_Recv(D, torcv, MPI_DOUBLE, MPI_ANY_SOURCE, 0, comm, &status);
  }
#if 0
  for (int i=0; i< count1; i++)
  {
    printf("Rank is %d and the value stored is %f \n",rank, D[i]);
  }
#endif


  if (coords[1] == 0)
    for (int i =0; i < count1 ; i++) local_x[i] = 0;
  int loop =0;
  double *w = (double*)malloc(count1*sizeof(double));
#if 0
  for (int i=0; i< count1; i++)
  {
    printf("---Rank is %d and the value is %f \n",rank, local_x[i]);
  }
#endif

  while (loop < max_iter)
  {
    distributed_matrix_vector_mult(n, R, local_x, w, comm);
    if(coords[1]==0)
      for (int i =0; i < count1; i++)
        local_x[i] = (local_b[i]-w[i])/D[i];
    distributed_matrix_vector_mult(n, local_A, local_x, w, comm);
    MPI_Barrier(comm);
    double sum = 0, val=0;
    if(coords[1]==0)
    {
      for( int i=0;i<count1;i++)
      {
        sum += ((local_b[i] - w[i])*(local_b[i] - w[i]));
      }
      //val = sqrt(sum);
    }
    MPI_Allreduce(&sum,&val, 1, MPI_DOUBLE, MPI_SUM, comm);
    val = sqrt(val);
    if(val <= l2_termination)
      break;
    loop++; 
  }
  free(R);
  free(D);
  free(w);
  free(col_vector);
#if 0
  if(coords[1]==0)
  {
  for (int i=0; i< count1; i++)
  {
    printf("*****Rank is %d and the value is %f \n",rank, local_x[i]);
  }}
#endif

}


// wraps the distributed matrix vector multiplication
void mpi_matrix_vector_mult(const int n, double* A,
                            double* x, double* y, MPI_Comm comm)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_x = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &x[0], &local_x, comm);
    

    // allocate local result space
    double* local_y = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_matrix_vector_mult(n, local_A, local_x, local_y, comm);

    // gather results back to rank 0
    gather_vector(n, local_y, y, comm);
}

// wraps the distributed jacobi function
void mpi_jacobi(const int n, double* A, double* b, double* x, MPI_Comm comm,
                int max_iter, double l2_termination)
{
    // distribute the array onto local processors!
    double* local_A = NULL;
    double* local_b = NULL;
    distribute_matrix(n, &A[0], &local_A, comm);
    distribute_vector(n, &b[0], &local_b, comm);

    // allocate local result space
    double* local_x = new double[block_decompose_by_dim(n, comm, 0)];
    distributed_jacobi(n, local_A, local_b, local_x, comm, max_iter, l2_termination);

    // gather results back to rank 0
    gather_vector(n, local_x, x, comm);
}
