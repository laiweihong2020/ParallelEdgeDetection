#include <mpi.h>
#include <stdio.h>

int main(int argc, char** argv) {
    MPI_Init(NULL, NULL);

    int world_size;
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);

    int world_rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &world_rank);

    if(world_rank == 0) {
        int number = 1;
        MPI_Ssend(&number, 1, MPI_INT, 1, 1, MPI_COMM_WORLD);
    }

    if(world_rank == 1) {
        int number = 2;
        MPI_Recv(&number, 1, MPI_INT, 0, 1, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("Received %d", number);
    }

    MPI_Finalize();
}