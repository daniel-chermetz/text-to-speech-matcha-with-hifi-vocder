const get_rounded_durations_per_token = `
	@group(0) @binding(0) var<storage, read_write> final_durations: array<u32>;
	@group(0) @binding(1) var<storage, read> durations: array<f32>;

	@group(0) @binding(2) var<storage, read> L_buffer: array<u32>;	
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let L: u32 = L_buffer[0];
		if (index >= L) {
			return;
		}

		final_durations[index] = u32(ceil(exp(durations[index])));
	}
`;

const calculate_cummulative_durations_up_to_each_position = `
	@group(0) @binding(0) var<storage, read_write> cummulative_durations: array<u32>;
	@group(0) @binding(1) var<storage, read> durations: array<u32>;

	@group(0) @binding(2) var<storage, read> L_buffer: array<u32>;	
	@compute @workgroup_size(1)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let L: u32 = L_buffer[0];
		
		cummulative_durations[0] = durations[0];
    
    	for (var i: u32 = 1u; i < L; i++) {
        	cummulative_durations[i] = cummulative_durations[i - 1] + durations[i];
    	}
	}
`;

const repeat_each_token_by_its_duration_val = `
	@group(0) @binding(0) var<storage, read_write> x_mel_dim_repeat_by_duration: array<f32>;
	@group(0) @binding(1) var<storage, read> x_mel_dim: array<f32>;
	@group(0) @binding(2) var<storage, read> cummulative_durations: array<u32>;

	@group(0) @binding(3) var<storage, read> L_buffer: array<u32>;	
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let L: u32 = L_buffer[0];
		
		if (index >= cummulative_durations[L - 1] * 80u) {
        	return;
    	}

    	let colIndex: u32 = index / 80u;
    	let rowIndex: u32 = index - colIndex * 80u;

    	var original_token_index = 0u;
    	for (var i: u32 = 0u; i < L; i++) {
	        if (colIndex < cummulative_durations[i]) {
	            original_token_index = i;
	            break;
	        }
		}

    	x_mel_dim_repeat_by_duration[index] = x_mel_dim[original_token_index * 80u + rowIndex];
	}
`;
