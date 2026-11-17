// mpishim: a tiny thread-based MPI subset for testing xpsim on machines
// without an MPI installation. Each "rank" is a thread of one process; the
// shim's main() spawns MPISHIM_NP (or -np N) ranks that each run xpsim_main().
// Only the calls xpsim needs are provided. NOT for performance measurements of
// communication -- use a real MPI for that.
#pragma once
#include <cstddef>

typedef int MPI_Comm;
typedef int MPI_Datatype;
typedef int MPI_Op;
typedef int MPI_Info;
typedef struct mpishim_req* MPI_Request;
typedef struct { int MPI_SOURCE, MPI_TAG, MPI_ERROR; } MPI_Status;
typedef void MPI_User_function(void*, void*, int*, MPI_Datatype*);

#define MPI_COMM_WORLD 0
#define MPI_COMM_NULL (-1)
#define MPI_BYTE 1
#define MPI_CHAR 2
#define MPI_INT 3
#define MPI_FLOAT 4
#define MPI_DOUBLE 5
#define MPI_UINT16_T 6
#define MPI_UNSIGNED_LONG_LONG 7
#define MPI_LONG_LONG 8
#define MPI_OP_NULL 0
#define MPI_SUM 1
#define MPI_MAX 2
#define MPI_MIN 3
#define MPI_LAND 4
#define MPI_IN_PLACE ((void*)1)
#define MPI_REQUEST_NULL ((MPI_Request)0)
#define MPI_STATUS_IGNORE ((MPI_Status*)0)
#define MPI_STATUSES_IGNORE ((MPI_Status*)0)
#define MPI_THREAD_SINGLE 0
#define MPI_THREAD_FUNNELED 1
#define MPI_THREAD_SERIALIZED 2
#define MPI_THREAD_MULTIPLE 3
#define MPI_SUCCESS 0
#define MPI_MAX_ERROR_STRING 256
#define MPI_TAG_UB 1
#define MPI_INFO_NULL 0
#define MPI_COMM_TYPE_SHARED 1

int MPI_Init_thread(int*, char***, int required, int* provided);
int MPI_Init(int*, char***);
int MPI_Finalize();
int MPI_Initialized(int*);
int MPI_Finalized(int*);
int MPI_Abort(MPI_Comm, int);
int MPI_Comm_rank(MPI_Comm, int*);
int MPI_Comm_size(MPI_Comm, int*);
int MPI_Comm_split(MPI_Comm, int color, int key, MPI_Comm*);
int MPI_Comm_split_type(MPI_Comm, int type, int key, MPI_Info, MPI_Comm*);
int MPI_Comm_dup(MPI_Comm, MPI_Comm*);
int MPI_Comm_free(MPI_Comm*);
int MPI_Comm_get_attr(MPI_Comm, int key, void* val, int* flag);
int MPI_Error_string(int, char*, int*);
int MPI_Op_create(MPI_User_function*, int commute, MPI_Op*);
int MPI_Op_free(MPI_Op*);
int MPI_Barrier(MPI_Comm);
int MPI_Bcast(void*, int, MPI_Datatype, int root, MPI_Comm);
int MPI_Allreduce(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm);
int MPI_Allgather(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm);
int MPI_Allgatherv(const void*, int, MPI_Datatype, void*, const int*, const int*, MPI_Datatype, MPI_Comm);
int MPI_Alltoallv(const void*, const int*, const int*, MPI_Datatype, void*, const int*, const int*, MPI_Datatype, MPI_Comm);
int MPI_Iallreduce(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm, MPI_Request*);
int MPI_Iallgather(const void*, int, MPI_Datatype, void*, int, MPI_Datatype, MPI_Comm, MPI_Request*);
int MPI_Ireduce_scatter_block(const void*, void*, int, MPI_Datatype, MPI_Op, MPI_Comm, MPI_Request*);
int MPI_Ibarrier(MPI_Comm, MPI_Request*);
int MPI_Isend(const void*, int, MPI_Datatype, int dst, int tag, MPI_Comm, MPI_Request*);
int MPI_Irecv(void*, int, MPI_Datatype, int src, int tag, MPI_Comm, MPI_Request*);
int MPI_Wait(MPI_Request*, MPI_Status*);
int MPI_Waitall(int, MPI_Request*, MPI_Status*);
int MPI_Test(MPI_Request*, int*, MPI_Status*);
int MPI_Testall(int, MPI_Request*, int*, MPI_Status*);
double MPI_Wtime();
