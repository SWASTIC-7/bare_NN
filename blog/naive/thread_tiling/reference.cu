
__global__ void tiled_thread_matmul(float *A, float *B, float *C, int N)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    const uint thread_id = threadIdx.x;

    const uint warp_id = thread_id / 32;
    const uint lane_id = thread_id % 32;

    const uint warp_row = warp_id / (TILE_WIDTH / WARP_TILE_WIDTH);
    const uint warp_col = warp_id % (TILE_WIDTH / WARP_TILE_WIDTH);

    const uint inner_row = warp_row * WARP_TILE_WIDTH + lane_id / WARP_TILE_WIDTH;
    const uint inner_col = warp_col * WARP_TILE_WIDTH + lane_id % WARP_TILE_WIDTH;

    int row = block_row * TILE_WIDTH + inner_row * THREAD_TILE_WIDTH;
    int col = block_col * TILE_WIDTH + inner_col * THREAD_TILE_WIDTH;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value[THREAD_TILE_WIDTH][THREAD_TILE_WIDTH] = {0};

    for (int i = 0; i < N / TILE_WIDTH; i++)
    {
        for (int r = 0; r < THREAD_TILE_WIDTH; r++)
            for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                sh_A[inner_row * THREAD_TILE_WIDTH + r][inner_col * THREAD_TILE_WIDTH + c] =
                    A[(row + r) * N + i * TILE_WIDTH + inner_col * THREAD_TILE_WIDTH + c];

        for (int r = 0; r < THREAD_TILE_WIDTH; r++)
            for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                sh_B[inner_row * THREAD_TILE_WIDTH + r][inner_col * THREAD_TILE_WIDTH + c] =
                    B[(i * TILE_WIDTH + inner_row * THREAD_TILE_WIDTH + r) * N + col + c];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            float reg_A[THREAD_TILE_WIDTH];
            float reg_B[THREAD_TILE_WIDTH];

            for (int r = 0; r < THREAD_TILE_WIDTH; r++)
                reg_A[r] = sh_A[inner_row * THREAD_TILE_WIDTH + r][k];

            for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                reg_B[c] = sh_B[k][inner_col * THREAD_TILE_WIDTH + c];

            for (int r = 0; r < THREAD_TILE_WIDTH; r++)
                for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                    value[r][c] += reg_A[r] * reg_B[c];
        }

        __syncthreads();
    }

    for (int r = 0; r < THREAD_TILE_WIDTH; r++)
        for (int c = 0; c < THREAD_TILE_WIDTH; c++)
            C[(row + r) * N + col + c] = value[r][c];
}

__global__ void tiled_thread_matmul(float *A, float *B, float *C, int N)
{
    const uint block_row = blockIdx.y;
    const uint block_col = blockIdx.x;

    const uint thread_id = threadIdx.x;

    const uint warp_id = thread_id / 32;
    const uint lane_id = thread_id % 32;

    const uint warp_row = warp_id / (TILE_WIDTH / WARP_TILE_WIDTH);
    const uint warp_col = warp_id % (TILE_WIDTH / WARP_TILE_WIDTH);

    const uint inner_row = warp_row * WARP_TILE_WIDTH + lane_id / WARP_TILE_WIDTH;
    const uint inner_col = warp_col * WARP_TILE_WIDTH + lane_id % WARP_TILE_WIDTH;

    int row = block_row * TILE_WIDTH + inner_row * THREAD_TILE_HEIGHT;
    int col = block_col * TILE_WIDTH + inner_col * THREAD_TILE_WIDTH;

    __shared__ float sh_A[TILE_WIDTH][TILE_WIDTH];
    __shared__ float sh_B[TILE_WIDTH][TILE_WIDTH];

    float value[THREAD_TILE_HEIGHT][THREAD_TILE_WIDTH] = {0};

    for (int i = 0; i < N / TILE_WIDTH; i++)
    {
        for (int r = 0; r < THREAD_TILE_HEIGHT; r++)
            for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                sh_A[inner_row * THREAD_TILE_HEIGHT + r][inner_col * THREAD_TILE_WIDTH + c] =
                    A[(row + r) * N + i * TILE_WIDTH + inner_col * THREAD_TILE_WIDTH + c];

        for (int r = 0; r < THREAD_TILE_HEIGHT; r++)
            for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                sh_B[inner_row * THREAD_TILE_HEIGHT + r][inner_col * THREAD_TILE_WIDTH + c] =
                    B[(i * TILE_WIDTH + inner_row * THREAD_TILE_HEIGHT + r) * N + col + c];

        __syncthreads();

        for (int k = 0; k < TILE_WIDTH; k++)
        {
            float reg_A[THREAD_TILE_HEIGHT];
            float reg_B[THREAD_TILE_WIDTH];

            for (int r = 0; r < THREAD_TILE_HEIGHT; r++)
                reg_A[r] = sh_A[inner_row * THREAD_TILE_HEIGHT + r][k];

            for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                reg_B[c] = sh_B[k][inner_col * THREAD_TILE_WIDTH + c];

            for (int r = 0; r < THREAD_TILE_HEIGHT; r++)
                for (int c = 0; c < THREAD_TILE_WIDTH; c++)
                    value[r][c] += reg_A[r] * reg_B[c];
        }

        __syncthreads();
    }

    for (int r = 0; r < THREAD_TILE_HEIGHT; r++)
        for (int c = 0; c < THREAD_TILE_WIDTH; c++)
            C[(row + r) * N + col + c] = value[r][c];
}