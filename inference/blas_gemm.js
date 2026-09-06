const matmul = `
	@group(0) @binding(0) var<storage, read_write> y: array<f32>;
	@group(0) @binding(1) var<storage, read> x1: array<f32>;
	@group(0) @binding(2) var<storage, read> x2: array<f32>;

	@group(0) @binding(3) var<storage, read> x1_rows_buffer: array<u32>;
	@group(0) @binding(4) var<storage, read> x2_rows_buffer: array<u32>;	
	@group(0) @binding(5) var<storage, read> x2_cols_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let x1_rows: u32 = x1_rows_buffer[0];
		let x2_rows: u32 = x2_rows_buffer[0];
		let x2_cols: u32 = x2_cols_buffer[0];
		
		if (index >= (x1_rows * x2_cols)) {
        	return;
    	}

		let targetCol: u32 = index / x1_rows;
		let targetRow: u32 = index - targetCol * x1_rows;

		let colOffset_x2: u32 = targetCol * x2_rows; 

		var sum: f32 = 0.0f;
    	for (var i: u32 = 0u; i < x2_rows; i++) {
			sum += (x1[targetRow + x1_rows * i] * x2[colOffset_x2 + i]);
    	}
		y[index] = sum;
	}
`;

const matmul_by_head_implicit_x1_transpose_no_fuse_target = `
	@group(0) @binding(0) var<storage, read_write> y: array<f32>;
	@group(0) @binding(1) var<storage, read> x1: array<f32>;
	@group(0) @binding(2) var<storage, read> x2: array<f32>;

	@group(0) @binding(3) var<storage, read> x1_rows_buffer: array<u32>;
	@group(0) @binding(4) var<storage, read> x2_rows_buffer: array<u32>;	

	@group(0) @binding(5) var<storage, read> num_heads_buffer: array<u32>;
	@group(0) @binding(6) var<storage, read> x1_head_rows_buffer: array<u32>;
	@group(0) @binding(7) var<storage, read> x1_head_cols_buffer: array<u32>;	
	@group(0) @binding(8) var<storage, read> x2_head_rows_buffer: array<u32>;		
	@group(0) @binding(9) var<storage, read> x2_head_cols_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;

		let x1_rows: u32 = x1_rows_buffer[0];
		let x2_rows: u32 = x2_rows_buffer[0];

		let num_heads: u32 = num_heads_buffer[0];
		let x1_head_rows: u32 = x1_head_rows_buffer[0];
		let x1_head_cols: u32 = x1_head_cols_buffer[0];
		let x2_head_rows: u32 = x2_head_rows_buffer[0];
		let x2_head_cols: u32 = x2_head_cols_buffer[0];	
		
		let num_head_elements: u32 = x1_head_cols * x2_head_cols; // before transpose x1 cols are its future rows

		if (index >= (num_heads * num_head_elements)) {
        	return;
    	}

		let targetHead: u32 = index / num_head_elements;
		let targetCol: u32 = index / x1_head_cols; // before transpose x1 cols are its future rows
		let targetHeadRelativeCol: u32 = targetCol - targetHead * x2_head_cols;
		let targetRow: u32 = index - targetCol * x1_head_cols;

		let originRowOffset_x1: u32 = targetHead * x1_head_rows;
		let originColOffset_x1: u32 = targetRow * x1_rows; // targetRow maps to x1 column

		let originRowOffset_x2: u32 = targetHead * x2_head_rows;
		let originColOffset_x2: u32 = targetHeadRelativeCol * x2_rows;

		let x1_offset: u32 = originColOffset_x1 + originRowOffset_x1;
		let x2_offset: u32 = originColOffset_x2 + originRowOffset_x2;

		var sum: f32 = 0.0f;
    	for (var i: u32 = 0u; i < x2_head_rows; i++) {
			sum += (x1[x1_offset + i] * x2[x2_offset + i]);
    	}
		y[index] = sum;
	}
`;

const matmul_by_head_x1_fused_x2_not_fused_y_fused = `
	@group(0) @binding(0) var<storage, read_write> y: array<f32>;
	@group(0) @binding(1) var<storage, read> x1: array<f32>;
	@group(0) @binding(2) var<storage, read> x2: array<f32>;

	@group(0) @binding(3) var<storage, read> x1_rows_buffer: array<u32>;

	@group(0) @binding(4) var<storage, read> x1_head_rows_buffer: array<u32>;
	@group(0) @binding(5) var<storage, read> x2_head_rows_buffer: array<u32>;		
	@group(0) @binding(6) var<storage, read> x2_head_cols_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;

		let x1_rows: u32 = x1_rows_buffer[0];

		let x1_head_rows: u32 = x1_head_rows_buffer[0];
		let x2_head_rows: u32 = x2_head_rows_buffer[0];
		let x2_head_cols: u32 = x2_head_cols_buffer[0];

		if (index >= (x1_rows * x2_head_cols)) {
        	return;
    	}

		let targetCol: u32 = index / x1_rows;
		let targetRow: u32 = index - x1_rows * targetCol;

		let headIndex: u32 = targetRow / x1_head_rows;
		let x2_headOffset: u32 = headIndex * x2_head_rows * x2_head_cols;
		let x2_colOffset = x2_headOffset + targetCol * x2_head_rows;

		var sum: f32 = 0.0f;
    	for (var i: u32 = 0u; i < x2_head_rows; i++) {
			sum += (x1[targetRow + x1_rows * i] * x2[x2_colOffset + i]);
    	}
		y[index] = sum;	
	}
`;
