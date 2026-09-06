const snakeBeta = `
    @group(0) @binding(0) var<storage, read_write> x: array<f32>;
    @group(0) @binding(1) var<storage, read> alpha: array<f32>; 
    @group(0) @binding(2) var<storage, read> beta: array<f32>;

    @group(0) @binding(3) var<storage, read> dim_buffer: array<u32>; 
    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>; 

    @compute @workgroup_size(32)
    fn main(
        @builtin(global_invocation_id) global_id: vec3<u32>
    ) {
        let index: u32 = global_id.x;
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];

        if (index >= dim * L) {
            return;
        }

        let colIndex: u32 = index / dim;
        let rowIndex: u32 = index - colIndex * dim;

        let val: f32 = x[index];
        let alpha_val: f32 = alpha[rowIndex] * val;
        let sin_alpha_val: f32 = sin(alpha_val);

        x[index] = val + (sin_alpha_val * sin_alpha_val) / (beta[rowIndex] + 1e-9f);        
    }
`;
