const set_initial_time_embedding = `
    @group(0) @binding(0) var<storage, read_write> time_embedding: array<f32>;
    @group(0) @binding(1) var<storage, read> f: array<f32>;

    @group(0) @binding(2) var<storage, read> t_buffer: array<f32>; 

    @compute @workgroup_size(32)
    fn main(
        @builtin(global_invocation_id) global_id: vec3<u32>
    ) {
        let index: u32 = global_id.x;
        let t: f32 = t_buffer[0];

        if (index >= 80u) {
            return;
        }

    	time_embedding[index] = sin(1000.0f * t * f[index]);
    	time_embedding[index + 80u] = cos(1000.0f * t * f[index]);        
    }
`;
