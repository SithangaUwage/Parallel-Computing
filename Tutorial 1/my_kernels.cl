//reduce using local memory + accumulation of local sums into a single location
//works with any number of groups - not optimal!
__kernel void reduce_add_4(__global const int* A, __global int* B, __local int* scratch) {
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2) {
		if (!(lid % (i * 2)) && ((lid + i) < N)) 
			scratch[lid] += scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid) {
		atomic_add(&B[0],scratch[lid]);
	}
}

// using reduce pattern to find minimum.

__kernel void reduce_find_min(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = (scratch[lid] < scratch[lid + i]) ? scratch[lid] : scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_min(&B[0], scratch[lid]);
	}
}

// using reduce pattern to find maximum.

__kernel void reduce_find_max(__global const int* A, __global int* B, __local int* scratch)
{
	int id = get_global_id(0);
	int lid = get_local_id(0);
	int N = get_local_size(0);

	//cache all N values from global memory to local memory
	scratch[lid] = A[id];

	barrier(CLK_LOCAL_MEM_FENCE);//wait for all local threads to finish copying from global to local memory

	for (int i = 1; i < N; i *= 2)
	{
		if (!(lid % (i * 2)) && ((lid + i) < N))
			scratch[lid] = (scratch[lid] > scratch[lid + i]) ? scratch[lid] : scratch[lid + i];

		barrier(CLK_LOCAL_MEM_FENCE);
	}

	//we add results from all local groups to the first element of the array
	//serial operation! but works for any group size
	//copy the cache to output array
	if (!lid)
	{
		atomic_max(&B[0], scratch[lid]);
	}
}
//a very simple histogram implementation
// olny work with the shorter set of data.
__kernel void hist_simple(__global const int* A, __global int* H) { 
	int id = get_global_id(0);

	//assumes that H has been initialised to 0
	int bin_index = A[id];//take value as a bin index

	atomic_inc(&H[bin_index]);//serial operation, not very efficient!
}

// bitonic sort to sort the data in ascending order

void cmpxchg(__global int* A, __global int* B, bool dir) {
	if ((!dir && *A > *B) || (dir && *A < *B)) {
		int t = *A;
		*A = *B;
		*B = t;
	}
}
void bitonic_merge(int id, __global int* A, int N, bool dir) {
	for (int i = N / 2; i > 0; i /= 2) {
		if ((id % (i * 2)) < i)
			cmpxchg(&A[id], &A[id + i], dir);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}

__kernel void sort_bitonic(__global int* A) {
	int id = get_global_id(0);
	int N = get_global_size(0);
	for (int i = 1; i < N / 2; i *= 2) {
		if (id % (i * 4) < i * 2)
			bitonic_merge(id, A, i * 2, false);
		else if ((id + i * 2) % (i * 4) < i * 2)
			bitonic_merge(id, A, i * 2, true);
		barrier(CLK_GLOBAL_MEM_FENCE);
	}
	bitonic_merge(id, A, N, false);
}

void cmpxchg1(__global int* A, __global int* B) {
	if (*A > *B) {
		int t = *A; *A = *B; *B = t;
	}
}

__kernel void OE_sort(__global int*A, __global int* B) {
	int id = get_global_id(0);
	int N = get_global_size(0);

	B[id] = A[id];

	for (int i = 0; i < N; i += 2) {
		if (id % 2 == 1 && id + 1 < N)
			cmpxchg1(&B[id], &B[id + 1]);

		barrier(CLK_GLOBAL_MEM_FENCE);

		if (id % 2 == 0 && id + 1 < N)
			cmpxchg1(&B[id], &B[id + 1]);

		barrier(CLK_GLOBAL_MEM_FENCE);
	}
}
