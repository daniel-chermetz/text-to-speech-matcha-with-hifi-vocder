const vocoder_concatAndOverlayTConv1D = `
	@group(0) @binding(0) var<storage, read_write> post: array<f32>;
	@group(0) @binding(1) var<storage, read> pre: array<f32>;
	@group(0) @binding(2) var<storage, read> bias: array<f32>;

	@group(0) @binding(3) var<storage, read> channels_buffer: array<u32>;
	@group(0) @binding(4) var<storage, read> timeSeriesSizePerChannel_buffer: array<u32>;
	@group(0) @binding(5) var<storage, read> preFrameCount_buffer: array<u32>;
	@group(0) @binding(6) var<storage, read> preFrameSize_buffer: array<u32>;
	@group(0) @binding(7) var<storage, read> preChannelSize_buffer: array<u32>;
	@group(0) @binding(8) var<storage, read> stride_buffer: array<u32>;
	@group(0) @binding(9) var<storage, read> padding_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let channels: u32 = channels_buffer[0];
		let timeSeriesSizePerChannel: u32 = timeSeriesSizePerChannel_buffer[0];
		let preFrameCount: u32 = preFrameCount_buffer[0];
		let preFrameSize: u32 = preFrameSize_buffer[0];
		let preChannelSize: u32 = preChannelSize_buffer[0];	
		let stride: u32 = stride_buffer[0];	
		let padding: u32 = padding_buffer[0];	

    	let maxCount: u32 = channels * timeSeriesSizePerChannel;
    	if (index >= maxCount) {
        	return;
    	}

    	let timeseriesIndex: u32 = index / channels; // post col
    	let channelIndex: u32 = index - timeseriesIndex * channels; // post row
    
    	let strideIndex: u32 = (timeseriesIndex + padding) / stride; // stride is 8 or 2
    	let strideRelativeIndex: u32 = (timeseriesIndex + padding) - strideIndex * stride;

    	// valid from strideIndex > 0
    	let preFrameIndex: i32 = i32(strideIndex) - 1i;
    	let preFrameFeatureIndex: u32 = stride + strideRelativeIndex;
    	let preFrameFeatureIndex2: u32 = strideRelativeIndex;

    	if (preFrameIndex < 0) {
        	post[index] = pre[preChannelSize * channelIndex + strideRelativeIndex] + bias[channelIndex];
    	} else if (u32(preFrameIndex) == preFrameCount - 1u) {
        	let frameOffset: u32 = u32(preFrameIndex) * preFrameSize;
        	let channelOffset: u32 = preChannelSize * channelIndex;
        	post[index] = pre[channelOffset + frameOffset + preFrameFeatureIndex] + bias[channelIndex];
    	} else {
        	let frameOffset: u32 = u32(preFrameIndex) * preFrameSize;
        	let frameOffset2: u32 = (u32(preFrameIndex) + 1u) * preFrameSize;
        	let channelOffset: u32 = preChannelSize * channelIndex;
        	post[index] = pre[channelOffset + frameOffset + preFrameFeatureIndex] + pre[channelOffset + frameOffset2 + preFrameFeatureIndex2] + bias[channelIndex];
    	}
	}
`;

const vocoder_adaptTensorToRightSideConvMul = `
	@group(0) @binding(0) var<storage, read_write> post: array<f32>;
	@group(0) @binding(1) var<storage, read> pre: array<f32>;

	@group(0) @binding(2) var<storage, read> originalChannels_buffer: array<u32>;
	@group(0) @binding(3) var<storage, read> originalSizePerChannel_buffer: array<u32>;
	@group(0) @binding(4) var<storage, read> k_buffer: array<u32>;
	@group(0) @binding(5) var<storage, read> dilation_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let originalChannels: u32 = originalChannels_buffer[0];
		let originalSizePerChannel: u32 = originalSizePerChannel_buffer[0];
		let k: u32 = k_buffer[0];
		let dilation: u32 = dilation_buffer[0];	

    	let maxCount: u32 = originalChannels * originalSizePerChannel * k;
    	if (index >= maxCount) {
        	return;
    	}

    	let postRows: u32 = originalChannels * k;
    	let colIndex: u32 = index / postRows;
    	let rowIndex: u32 = index - colIndex * postRows;

    	let originalChannelIndex: u32 = rowIndex / k;
    	let featureRelativeToOriginalChannel: u32 = rowIndex - originalChannelIndex * k;

    	// dilation=2, k=7: -6, -4, -2, 0, 2, 4, 6
    	// dilation=2, k=7: 4, 6, 8, 10, 12, 14, 16    
    	let preColIndex: i32 = i32(colIndex) + i32(featureRelativeToOriginalChannel) * i32(dilation) - (i32(k) / 2i) * i32(dilation);
    	if (preColIndex < 0i || preColIndex >= i32(originalSizePerChannel)) {
        	post[index] = 0.0f;
        	return;
    	}
    
    	let preIndex: u32 = originalChannelIndex + u32(preColIndex) * originalChannels;
    	post[index] = pre[preIndex];    		
	}
`;

const vocoder_leakyReLU = `
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

const vocoder_elementwiseAdd = `
	@group(0) @binding(0) var<storage, read_write> x1: array<f32>;
	@group(0) @binding(1) var<storage, read> x2: array<f32>;

	@group(0) @binding(2) var<storage, read> maxCount_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let maxCount: u32 = maxCount_buffer[0];

		if (index >= maxCount) {
			return;
		}

    	x1[index] += x2[index];
	}
`;

const vocoder_elementwiseAddAverageThreeTensors = `
	@group(0) @binding(0) var<storage, read_write> x1: array<f32>;
	@group(0) @binding(1) var<storage, read> x2: array<f32>;
	@group(0) @binding(2) var<storage, read> x3: array<f32>;

	@group(0) @binding(3) var<storage, read> maxCount_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let maxCount: u32 = maxCount_buffer[0];

		if (index >= maxCount) {
			return;
		}

    	x1[index] = ((x1[index] + x2[index] + x3[index]) / 3.0f);
	}
`;

const vocoder_apply_tanh = `
	@group(0) @binding(0) var<storage, read_write> x: array<f32>;

	@group(0) @binding(1) var<storage, read> maxCount_buffer: array<u32>;

	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let maxCount: u32 = maxCount_buffer[0];

		if (index >= maxCount) {
			return;
		}

    	x[index] = tanh(x[index]);
	}
`;

const vocoder_add_bias = `
	@group(0) @binding(0) var<storage, read_write> tensor: array<f32>;
	@group(0) @binding(1) var<storage, read> bias: array<f32>;

	@group(0) @binding(2) var<storage, read> numChannels_buffer: array<u32>;
	@group(0) @binding(3) var<storage, read> maxCount_buffer: array<u32>;
	@compute @workgroup_size(32)
	fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
		let index: u32 = global_id.x;
		let numChannels: u32 = numChannels_buffer[0];
		let maxCount: u32 = maxCount_buffer[0];

		if (index >= maxCount) {
			return;
		}

    	let colIndex: u32 = index / numChannels;
    	let channelIndex: u32 = index - colIndex * numChannels;

    	tensor[index] += bias[channelIndex];		
	}
`;
