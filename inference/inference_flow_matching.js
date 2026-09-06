const concat_duration_repeated_tokens_to_noise = `
    @group(0) @binding(0) var<storage, read_write> combined: array<f32>;
    @group(0) @binding(1) var<storage, read> tokens_by_durations: array<f32>;
    @group(0) @binding(2) var<storage, read> noise: array<f32>;

    @group(0) @binding(3) var<storage, read> frameCount_buffer: array<u32>;
    @group(0) @binding(4) var<storage, read> compatibleFrameCount_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let frameCount: u32 = frameCount_buffer[0];
        let compatibleFrameCount: u32 = compatibleFrameCount_buffer[0];
        
        if (index >= compatibleFrameCount * 160u) {
            return;
        }

        let colIndex: u32 = index / 160;
        if (colIndex >= frameCount) {
            combined[index] = 0.0f;
            return;
        }

        let rowIndex: u32 = index - colIndex * 160u;

        if (rowIndex < 80u) {
            combined[index] = noise[colIndex * 80u + rowIndex];
        } else {
            combined[index] = tokens_by_durations[colIndex * 80u + rowIndex - 80u];
        }
    }
`;

const applySilu = `
    @group(0) @binding(0) var<storage, read_write> x: array<f32>;

    @group(0) @binding(1) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(2) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];
        
        if (index >= dim * L) {
            return;
        }

        // SiLU: x / (1 + exp(-x))
        let val: f32 = x[index];
        x[index] = val / (1.0f + exp(-val));
    }
`;

const mask_compatiblity_cols = `
    @group(0) @binding(0) var<storage, read_write> x: array<f32>;

    @group(0) @binding(1) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(2) var<storage, read> mel_frame_count_buffer: array<u32>;  
    @group(0) @binding(3) var<storage, read> compatible_mel_frame_count_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let mel_frame_count: u32 = mel_frame_count_buffer[0];
        let compatible_mel_frame_count: u32 = compatible_mel_frame_count_buffer[0];

        if (index >= ((compatible_mel_frame_count - mel_frame_count) * dim)) {
            return;
        }

        let offset: u32 = mel_frame_count * dim;
        x[offset + index] = 0.0f;
    }
`;

const get_group_mean = `
    @group(0) @binding(0) var<storage, read_write> meanByGroup: array<f32>;    
    @group(0) @binding(1) var<storage, read> x: array<f32>;

    @group(0) @binding(2) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(3) var<storage, read> groupDim_buffer: array<u32>;  
    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let groupIndex: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let groupDim: u32 = groupDim_buffer[0];
        let L: u32 = L_buffer[0];

        if (groupIndex >= (dim / groupDim)) {
            return;
        }

        let rowOffset: u32 = groupIndex * groupDim;
        var mean: f32 = 0f;
        for (var col: u32 = 0u; col < L; col++) {
            for (var row: u32 = rowOffset; row < (rowOffset + groupDim); row++) {
                mean += x[col * dim + row];
            }
        }

        meanByGroup[groupIndex] = mean / f32(groupDim * L);
    }
`;

const get_group_variance = `
    @group(0) @binding(0) var<storage, read_write> varianceByGroup: array<f32>;
    @group(0) @binding(1) var<storage, read> meanByGroup: array<f32>;    
    @group(0) @binding(2) var<storage, read> x: array<f32>;

    @group(0) @binding(3) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(4) var<storage, read> groupDim_buffer: array<u32>;  
    @group(0) @binding(5) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(8)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let groupIndex: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let groupDim: u32 = groupDim_buffer[0];
        let L: u32 = L_buffer[0];

        if (groupIndex >= (dim / groupDim)) {
            return;
        }

        let groupMean: f32 = meanByGroup[groupIndex];
        let rowOffset: u32 = groupIndex * groupDim;

        var variance: f32 = 0f;
        for (var col: u32 = 0u; col < L; col++) {
            for (var row: u32 = rowOffset; row < (rowOffset + groupDim); row++) {
                let v: f32 = (x[col * dim + row] - groupMean);
                variance += (v * v);
            }
        }

        varianceByGroup[groupIndex] = variance / f32(groupDim * L);
    }
`;

const group_norm = `
    @group(0) @binding(0) var<storage, read_write> x: array<f32>;
    @group(0) @binding(1) var<storage, read> varianceByGroup: array<f32>;
    @group(0) @binding(2) var<storage, read> meanByGroup: array<f32>;    
    @group(0) @binding(3) var<storage, read> gammaBias: array<f32>;    
    @group(0) @binding(4) var<storage, read> betaBias: array<f32>;    

    @group(0) @binding(5) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(6) var<storage, read> groupDim_buffer: array<u32>;  
    @group(0) @binding(7) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let groupDim: u32 = groupDim_buffer[0];
        let L: u32 = L_buffer[0];

        if (index >= (dim * L)) {
            return;
        }

        let colIndex: u32 = index / dim;
        let featureIndex: u32 = index - colIndex * dim;
        let groupIndex: u32 = featureIndex / groupDim;

        x[index] = gammaBias[featureIndex] 
            * (x[index] - meanByGroup[groupIndex]) 
            * inverseSqrt(varianceByGroup[groupIndex] + 1e-5f) 
            + betaBias[featureIndex];
    }
`;

const applyMish = `
    @group(0) @binding(0) var<storage, read_write> x: array<f32>;   

    @group(0) @binding(1) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(2) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];

        if (index >= (dim * L)) {
            return;
        }

        let val: f32 = x[index];

        var softplus: f32 = val;
        if (val <= 20.0f) {
            softplus = log(1.0f + exp(val));
        }
        x[index] = val * tanh(softplus);
    }
`;

const elementwiseAdd = `
    @group(0) @binding(0) var<storage, read_write> x1: array<f32>;   
    @group(0) @binding(1) var<storage, read> x2: array<f32>;   

    @group(0) @binding(2) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(3) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];

        if (index >= (dim * L)) {
            return;
        }

        x1[index] += x2[index];
    }
`;

const concat_to_double_rows_per_col = `
    @group(0) @binding(0) var<storage, read_write> combined: array<f32>;    
    @group(0) @binding(1) var<storage, read> x1: array<f32>;   
    @group(0) @binding(2) var<storage, read> x2: array<f32>;   

    @group(0) @binding(3) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];
        let twiceDim = 2u * dim;

        if (index >= (twiceDim * L)) {
            return;
        }

        let colIndex: u32 = index / twiceDim;
        let rowIndex: u32 = index - colIndex * twiceDim;

        if (rowIndex < dim) {
            combined[index] = x1[colIndex * dim + rowIndex];
        } else {
            combined[index] = x2[colIndex * dim + rowIndex - dim];
        }
    }
`;

// not fully general, but good for the current use case
const attach_splats_conv_transpose_1D = `
    @group(0) @binding(0) var<storage, read_write> x: array<f32>;    
    @group(0) @binding(1) var<storage, read> splats: array<f32>;   

    @group(0) @binding(2) var<storage, read> kernel_buffer: array<u32>;
    @group(0) @binding(3) var<storage, read> stride_buffer: array<u32>;
    @group(0) @binding(4) var<storage, read> padding_buffer: array<u32>;  
    @group(0) @binding(5) var<storage, read> targetRows_buffer: array<u32>;  
    @group(0) @binding(6) var<storage, read> originCols_buffer: array<u32>;  
    @group(0) @binding(7) var<storage, read> targetCols_buffer: array<u32>;  


    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let kernel: u32 = kernel_buffer[0];
        let stride: u32 = stride_buffer[0];
        let padding: u32 = padding_buffer[0];
        let targetRows: u32 = targetRows_buffer[0];
        let originCols: u32 = originCols_buffer[0];
        let targetCols: u32 = targetCols_buffer[0];

        if (index >= (targetRows * targetCols)) {
            return;
        }

        let targetCol: u32 = index / targetRows;
        let targetRow: u32 = index - targetCol * targetRows;

        let targetColWithOffset: i32 = i32(targetCol) + i32(padding) - i32(stride);
        let targetRowOffsetInSplats: u32 = targetRow * kernel;

        if (targetColWithOffset < 0) {
            // 0 target col + 1 padding 
            x[index] = splats[
                targetRowOffsetInSplats * originCols + 
                (targetCol + padding) * originCols
            ];
            return;
        }

        let leftSplatIndex: u32 = u32(targetColWithOffset) / stride; // 2/2=1, 3/2=1, 4/2=2,...
        // 0 for the 3rd of left and 1st of right
        // 1 for the 4th of left and 2nd of right
        let withinSplatIndex: u32 = u32(targetColWithOffset) - leftSplatIndex * stride;

        if (leftSplatIndex == (originCols - 1)) {
            x[index] = 
                splats[leftSplatIndex + (stride + withinSplatIndex) * originCols + targetRowOffsetInSplats * originCols];
        } else {
            x[index] = 
                splats[leftSplatIndex + (stride + withinSplatIndex) * originCols + targetRowOffsetInSplats * originCols] +
                splats[leftSplatIndex + 1 + (withinSplatIndex) * originCols + targetRowOffsetInSplats * originCols];
        }
    }
`;

const add_velocity_step_to_evolving_noise = `
    @group(0) @binding(0) var<storage, read_write> evolving_x_noise: array<f32>;    
    @group(0) @binding(1) var<storage, read> velocity: array<f32>;   

    @group(0) @binding(2) var<storage, read> delta_time_buffer: array<f32>;
    @group(0) @binding(3) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let delta_time: f32 = delta_time_buffer[0];    
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];

        if (index >= (dim * L)) {
            return;
        }

        evolving_x_noise[index] += (velocity[index] * delta_time);
    }
`;

const get_final_user_facing_mel = `
    @group(0) @binding(0) var<storage, read_write> evolving_x_noise: array<f32>;    

    @group(0) @binding(1) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(2) var<storage, read> L_buffer: array<u32>;  

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];

        if (index >= (dim * L)) {
            return;
        }

        evolving_x_noise[index] = 2.116101f * evolving_x_noise[index] - 5.536622f;
    }
`;
