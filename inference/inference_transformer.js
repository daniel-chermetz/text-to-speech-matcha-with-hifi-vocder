const applyRoPE = `
    @group(0) @binding(0) var<storage, read_write> keysOrQueries: array<f32>;
    @group(0) @binding(1) var<storage, read> preComputedRopeTheta: array<f32>;

    @group(0) @binding(2) var<storage, read> dim_buffer: array<u32>;
    @group(0) @binding(3) var<storage, read> headDim_buffer: array<u32>;    
    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>;  
    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let headDim: u32 = headDim_buffer[0];
        let dim: u32 = dim_buffer[0];
        let L: u32 = L_buffer[0];
        if (index >= dim * L) {
            return;
        }

       let halfHeadDim: u32 = headDim / 2;
       let quarterHeadDim: u32 = halfHeadDim / 2;

        let colIndex: u32 = index / dim;
        let rowIndex: u32 = index - colIndex * dim;
        let headIndex: u32 = rowIndex / headDim;
        let headRelativeRowIndex: u32 = rowIndex - headIndex * headDim;

        if (headRelativeRowIndex >= halfHeadDim) {
            return;
        }
        if (headRelativeRowIndex >= quarterHeadDim) {
            return;
        }

        let headRelativeColOffset: u32 = colIndex * headDim;
        let cosThetaIndex: u32 = headRelativeColOffset + headRelativeRowIndex * 2;
        let cosTheta: f32 = preComputedRopeTheta[cosThetaIndex]; // up to 46
        let sinTheta: f32 = preComputedRopeTheta[cosThetaIndex + 1]; // up to 47

        let firstValOfPair: f32 = keysOrQueries[index];
        let secondPairIndex: u32 = index + quarterHeadDim;
        let secondValOfPair: f32 = keysOrQueries[secondPairIndex];
        keysOrQueries[index] = cosTheta * firstValOfPair - sinTheta * secondValOfPair;
        keysOrQueries[secondPairIndex] = sinTheta * firstValOfPair + cosTheta * secondValOfPair;
    }
`;


const getHeadDimScaledAttn = `
    @group(0) @binding(0) var<storage, read_write> attnKtQByHead: array<f32>;

    @group(0) @binding(1) var<storage, read> headDimScale_buffer: array<f32>;
    @group(0) @binding(2) var<storage, read> numAttnHeads_buffer: array<u32>;    
    @group(0) @binding(3) var<storage, read> L_valid_buffer: array<u32>;
    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>; 

    @compute @workgroup_size(32)
    fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
        let index: u32 = global_id.x;
        let headDimScale: f32 = headDimScale_buffer[0];
        let numAttnHeads: u32 = numAttnHeads_buffer[0];
        let L_valid: u32 = L_valid_buffer[0];
        let L: u32 = L_buffer[0];

        let L2: u32 = L * L;
        if (index >= numAttnHeads * L2) {
            return;
        }

        let colIndex: u32 = index / L;
        let headIndex: u32 = colIndex / L;
        let headRelativeColIndex: u32 = colIndex - headIndex * L;
        let headRelativeRowIndex: u32 = index - headIndex * L2 - headRelativeColIndex * L;
        if (headRelativeRowIndex >= L_valid) {
            attnKtQByHead[index] = -1e9f;
            return;       
        }

        attnKtQByHead[index] = attnKtQByHead[index] * headDimScale;
    }
`;

const getAttnHeadsMaxByCol_softmax = `
    @group(0) @binding(0) var<storage, read_write> attnByHead_maxByCol_softmax: array<f32>;
    @group(0) @binding(1) var<storage, read> attnHeadDimScaledMaskedKtQByHead: array<f32>;

    @group(0) @binding(2) var<storage, read> L_buffer: array<u32>; 

    var<workgroup> sData: array<f32, 32>;

    @compute @workgroup_size(32)
    fn main(
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
        let attnHeadIndex: u32 = workgroup_id.x;
        let headRelativeColIndex: u32 = workgroup_id.y;
        let tIndex: u32 = local_id.x;
        let L = L_buffer[0];

        let colIndex: u32 = attnHeadIndex * L + headRelativeColIndex;
        let colOffset: u32 = colIndex * L;

        var threadColMax: f32 = -1.0e20f;
        for (var rowIndex: u32 = tIndex; rowIndex < L; rowIndex += 32u) {
            if (attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex] > threadColMax) {
                threadColMax = attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex];
            }
        }
        sData[tIndex] = threadColMax;
        workgroupBarrier();

        for (var reductionSize: u32 = 16u; reductionSize > 0; reductionSize /= 2) {
            if ((tIndex < reductionSize) && (sData[tIndex] < sData[tIndex + reductionSize])) {
                sData[tIndex] = sData[tIndex + reductionSize];
            }
            workgroupBarrier();
        }

        if (tIndex == 0) {
            attnByHead_maxByCol_softmax[colIndex] = sData[0];
        }
    }
`;

const getAttnHeadsSumByCol_softmax = `
    @group(0) @binding(0) var<storage, read_write> attnByHead_sumByCol_softmax: array<f32>;
    @group(0) @binding(1) var<storage, read_write> attnByHead_expfCache_softmax: array<f32>;
    @group(0) @binding(2) var<storage, read> attnHeadDimScaledMaskedKtQByHead: array<f32>;
    @group(0) @binding(3) var<storage, read> attnByHead_maxByCol_softmax: array<f32>;

    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>; 

    var<workgroup> sData: array<f32, 32>;

    @compute @workgroup_size(32)
    fn main(
        @builtin(workgroup_id) workgroup_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>
    ) {
        let attnHeadIndex: u32 = workgroup_id.x;
        let headRelativeColIndex: u32 = workgroup_id.y;
        let tIndex: u32 = local_id.x;
        let L = L_buffer[0];

        let colIndex: u32 = attnHeadIndex * L + headRelativeColIndex;
        let colOffset: u32 = colIndex * L;

        var threadSum: f32 = 0.0f;
        for (var rowIndex: u32 = tIndex; rowIndex < L; rowIndex += 32u /* numThreads per block */) {
            let expVal: f32 = exp(attnHeadDimScaledMaskedKtQByHead[colOffset + rowIndex] - attnByHead_maxByCol_softmax[colIndex]);
            attnByHead_expfCache_softmax[colOffset + rowIndex] = expVal;
            threadSum += expVal;
        }
        sData[tIndex] = threadSum;
        workgroupBarrier();

        // blockDim.x = 32; reductionSize = 16; reductionSize > 0; reductionSize: 16, 8, 4, etc.
        for (var reductionSize: u32 = 16u; reductionSize > 0; reductionSize /= 2) {
            if (tIndex < reductionSize) {
                sData[tIndex] = sData[tIndex] + sData[tIndex + reductionSize];
            }
            workgroupBarrier();
        }

        if (tIndex == 0) {
            attnByHead_sumByCol_softmax[colIndex] = sData[0];
        }
    }
`;

const applySoftmaxToAttnHeads = `
    @group(0) @binding(0) var<storage, read_write> attnByHead_postSoftmax: array<f32>;
    @group(0) @binding(1) var<storage, read> expfCache: array<f32>;
    @group(0) @binding(2) var<storage, read> sumByCol: array<f32>;

    @group(0) @binding(3) var<storage, read> numAttnHeads_buffer: array<u32>; 
    @group(0) @binding(4) var<storage, read> L_buffer: array<u32>; 

    @compute @workgroup_size(32)
    fn main(
        @builtin(global_invocation_id) global_id: vec3<u32>
    ) {
        let index: u32 = global_id.x;
        let numAttnHeads: u32 = numAttnHeads_buffer[0];
        let L: u32 = L_buffer[0];

        let L2: u32 = L * L;
        if (index >= (numAttnHeads * L2)) {
            return;
        }

        let headIndex: u32 = index / L2;
        let headRelativeColIndex: u32 = (index - headIndex * L2) / L;
        let globalColIndex: u32 = headIndex * L + headRelativeColIndex;

        attnByHead_postSoftmax[index] = (expfCache[index] / sumByCol[globalColIndex]);
    }
`;
