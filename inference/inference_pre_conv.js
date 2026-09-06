const set_x_embeddings = `
	@group(0) @binding(0) var<storage, read_write> x: array<f32>;
	@group(0) @binding(1) var<storage, read> embedding_weights: array<f32>;
	@group(0) @binding(2) var<storage, read> token_seq_indices: array<u32>;

	@group(0) @binding(3) var<storage, read> dim_buffer: array<u32>;
	@group(0) @binding(4) var<storage, read> L_buffer: array<u32>;	
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let dim: u32 = dim_buffer[0];
		let L: u32 = L_buffer[0];
		if (index >= dim * L) {
			return;
		}

		let colIndex: u32 = index / dim;
		let tokenIndex: u32 = token_seq_indices[colIndex];
		let featureIndex: u32 = index - colIndex * dim;

		x[index] = embedding_weights[tokenIndex * dim + featureIndex] * sqrt(f32(dim));
	}
`;

const im2col_prepare_for_1D_conv = `
	@group(0) @binding(0) var<storage, read_write> x_col: array<f32>;
	@group(0) @binding(1) var<storage, read> x: array<f32>;

	@group(0) @binding(2) var<storage, read> kernel_buffer: array<u32>;	
	@group(0) @binding(3) var<storage, read> padding_buffer: array<u32>;
	@group(0) @binding(4) var<storage, read> stride_buffer: array<u32>;
	@group(0) @binding(5) var<storage, read> dim_buffer: array<u32>;
	@group(0) @binding(6) var<storage, read> L_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let kernel: u32 = kernel_buffer[0];
		let padding: u32 = padding_buffer[0];
		let stride: u32 = stride_buffer[0];
		let dim: u32 = dim_buffer[0];
		let L: u32 = L_buffer[0];

		let targetL: u32 = (L + 2 * padding - kernel) / stride + 1;
		let targetColSize: u32 = dim * kernel; // 960
		
		if (index >= targetColSize * targetL) {
			return;
		}

		let targetColIndex: u32 = index / targetColSize; // 1803 / 960 = 1
		let originColIndex: i32 = i32(targetColIndex * stride) - i32(padding); // zeroth column of current kernel, -1

		let targetFeatureIndex: u32 = index - targetColIndex * targetColSize; // 1803 - 960 = 843
		let originFeatureIndex: u32 = targetFeatureIndex / kernel; // 843 / 5 = 168
		let originColRelativeTokenIndex: u32 = targetFeatureIndex - originFeatureIndex * kernel; // 843 - 168 * 5 = 3
		let originFinalColIndex: i32 = originColIndex + i32(originColRelativeTokenIndex); // -1 + 3 = 2
		
		if (originFinalColIndex < 0 || originFinalColIndex >= i32(L)) {
			x_col[index] = 0.0f;
			return;
		}
		x_col[index] = x[u32(originFinalColIndex) * dim + originFeatureIndex];
	}
`;

const add_bias = `
	@group(0) @binding(0) var<storage, read_write> x: array<f32>;
	@group(0) @binding(1) var<storage, read> bias: array<f32>;

	@group(0) @binding(2) var<storage, read> dim_buffer: array<u32>;
	@group(0) @binding(3) var<storage, read> L_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let dim: u32 = dim_buffer[0];
		let L: u32 = L_buffer[0];

		if (index >= dim * L) {
			return;
		}

		let col: u32 = index / dim;
		let feature: u32 = index - col * dim;

		x[index] += bias[feature];		
	}
`;

const get_col_mean = `
	@group(0) @binding(0) var<storage, read_write> mean_by_col: array<f32>;
	@group(0) @binding(1) var<storage, read> x: array<f32>;

	@group(0) @binding(2) var<storage, read> dim_buffer: array<u32>;
	@group(0) @binding(3) var<storage, read> L_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let dim: u32 = dim_buffer[0];
		let L: u32 = L_buffer[0];

		if (index >= L) {
			return;
		}

		let colOffset: u32 = index * dim;

		var sum: f32 = 0;
		for (var i: u32 = 0; i < dim; i++) {
			sum += x[colOffset + i];
		}

		mean_by_col[index] = sum / f32(dim);		
	}
`;

const get_col_variance = `
	@group(0) @binding(0) var<storage, read_write> variance_by_col: array<f32>;
	@group(0) @binding(1) var<storage, read> mean_by_col: array<f32>;	
	@group(0) @binding(2) var<storage, read> x: array<f32>;

	@group(0) @binding(3) var<storage, read> dim_buffer: array<u32>;
	@group(0) @binding(4) var<storage, read> L_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let dim: u32 = dim_buffer[0];
		let L: u32 = L_buffer[0];

		if (index >= L) {
			return;
		}

		let colOffset: u32 = index * dim;

		var variance: f32 = 0;
		for (var i: u32 = 0; i < dim; i++) {
			let d: f32 = x[colOffset + i] - mean_by_col[index];
			variance += (d * d);
		}

		variance_by_col[index] = variance / f32(dim);		
	}
`;

const layer_norm = `
	@group(0) @binding(0) var<storage, read_write> x: array<f32>;
	@group(0) @binding(1) var<storage, read> variance_by_col: array<f32>;
	@group(0) @binding(2) var<storage, read> mean_by_col: array<f32>;
	@group(0) @binding(3) var<storage, read> gammaBias: array<f32>;
	@group(0) @binding(4) var<storage, read> betaBias: array<f32>;

	@group(0) @binding(5) var<storage, read> eps_buffer: array<f32>;
	@group(0) @binding(6) var<storage, read> dim_buffer: array<u32>;
	@group(0) @binding(7) var<storage, read> L_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let eps: f32 = eps_buffer[0];
		let dim: u32 = dim_buffer[0];
		let L: u32 = L_buffer[0];

		if (index >= dim * L) {
			return;
		}

		let colIndex: u32 = index / dim;
		let featureIndex: u32 = index - colIndex * dim;

		x[index] = gammaBias[featureIndex] * (x[index] - mean_by_col[colIndex]) * inverseSqrt(variance_by_col[colIndex] + eps) + betaBias[featureIndex];		
	}
`;

const leakyReLU = `
	@group(0) @binding(0) var<storage, read_write> x: array<f32>;	

	@group(0) @binding(1) var<storage, read> slope_buffer: array<f32>;
	@group(0) @binding(2) var<storage, read> maxCount_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let slope: f32 = slope_buffer[0];
		let maxCount: u32 = maxCount_buffer[0];

		if (index >= maxCount) {
			return;
		}

	    var postVal: f32 = x[index];
	    if (postVal < 0) {
	        postVal = postVal * slope;
	    }
	    x[index] = postVal;		
	}
`;

const add_residual = `
	@group(0) @binding(0) var<storage, read_write> x: array<f32>;	
	@group(0) @binding(1) var<storage, read> residual: array<f32>;	

	@group(0) @binding(2) var<storage, read> dim_buffer: array<u32>;
	@group(0) @binding(3) var<storage, read> L_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let dim: u32 = dim_buffer[0];
		let L: u32 = L_buffer[0];

		if (index >= dim * L) {
			return;
		}

		x[index] += residual[index];	
	}
`;
