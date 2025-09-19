let inputData = [];
let outputData = [];
let currentNumRows = 2;
let currentFeatureDim = 8;
let currentBlockSize = 4;
let isRunning = false;
let maxValues = [];
let sumExpValues = [];

function initializeData() {
    const prevFeatureDim = currentFeatureDim;
    const prevNumRows = currentNumRows;

    currentNumRows = parseInt(document.getElementById('numRows').value);
    currentFeatureDim = parseInt(document.getElementById('featureDim').value);

    // Update block size options based on feature dimension
    updateBlockSizeOptions();
    currentBlockSize = parseInt(document.getElementById('blockSize').value);

    // Only regenerate input data if dimensions changed
    const dimensionsChanged = (prevFeatureDim !== currentFeatureDim || prevNumRows !== currentNumRows);

    if (dimensionsChanged || !inputData || inputData.length === 0) {
        // Initialize 2D arrays for input data
        inputData = [];
        for (let row = 0; row < currentNumRows; row++) {
            inputData[row] = generatePresetInput(currentFeatureDim, row);
        }
    }

    // Always reset output data and calculations
    outputData = [];
    maxValues = [];
    sumExpValues = [];
    for (let row = 0; row < currentNumRows; row++) {
        outputData[row] = new Array(currentFeatureDim).fill(0);
        maxValues[row] = 0;
        sumExpValues[row] = 0;
    }

    updateDisplays();
}

function generatePresetInput(dim, rowIndex) {
    // Use the exact values from the blog post example
    const exampleRows = [
        [2.0, -1.0, 3.0, 0.5, -0.5, 1.5, -2.0, 1.0], // Row 0
        [4.0, -3.0, 2.5, 1.0, -1.5, 0.0, -0.5, 2.0], // Row 1
        [-1.0, 3.5, -2.5, 1.5, 0.0, -3.0, 2.5, -0.5]  // Row 2
    ];

    if (rowIndex < exampleRows.length && dim <= 8) {
        // Use exact values from example, truncated to current dimension
        return exampleRows[rowIndex].slice(0, dim);
    }

    // Fallback for additional rows or dimensions
    const preset = [];
    const rowVariation = (rowIndex + 1) * 0.8;

    for (let i = 0; i < dim; i++) {
        if (rowIndex % 2 === 0) {
            preset[i] = rowVariation * (2.0 + Math.sin(i * 0.7)) * (i % 2 === 0 ? 1 : -0.7);
        } else {
            preset[i] = rowVariation * (1.8 + Math.cos(i * 0.9)) * (i % 3 === 0 ? -1 : 0.9);
        }
    }
    return preset;
}

function generateRandomInput() {
    if (isRunning) return;

    for (let row = 0; row < currentNumRows; row++) {
        for (let i = 0; i < currentFeatureDim; i++) {
            inputData[row][i] = (Math.random() - 0.5) * 8; // Random values between -4 and 4
        }
        outputData[row].fill(0);
    }
    updateDisplays();
    reset();
}

function updateDisplays() {
    updateTensorDisplay('inputTensor', inputData, true);
    updateTensorDisplay('outputTensor', outputData, false);
    updateCalculationsTable();
    updateBlockSizeDisplay();
}

function updateTensorDisplay(elementId, data, isInput) {
    const container = document.getElementById(elementId);
    const cols = currentFeatureDim;
    const rows = currentNumRows;

    container.style.gridTemplateColumns = `repeat(${cols}, 1fr)`;
    container.innerHTML = '';

    for (let row = 0; row < rows; row++) {
        for (let col = 0; col < cols; col++) {
            const cell = document.createElement('div');
            cell.className = 'tensor-cell';
            cell.textContent = data[row][col].toFixed(3);
            cell.id = `${elementId}_${row}_${col}`;
            container.appendChild(cell);
        }
    }
}

function highlightRow(rowIdx, className) {
    for (let col = 0; col < currentFeatureDim; col++) {
        const inputCell = document.getElementById(`inputTensor_${rowIdx}_${col}`);
        const outputCell = document.getElementById(`outputTensor_${rowIdx}_${col}`);
        if (inputCell) {
            inputCell.className = `tensor-cell ${className}`;
        }
        if (outputCell) {
            outputCell.className = `tensor-cell ${className}`;
        }
    }
}

function highlightBlock(rowIdx, startIdx, endIdx, className) {
    for (let i = startIdx; i < Math.min(endIdx, currentFeatureDim); i++) {
        const inputCell = document.getElementById(`inputTensor_${rowIdx}_${i}`);
        const outputCell = document.getElementById(`outputTensor_${rowIdx}_${i}`);
        if (inputCell) {
            inputCell.className = `tensor-cell ${className}`;
        }
        if (outputCell) {
            outputCell.className = `tensor-cell ${className}`;
        }
    }
}

function clearHighlights() {
    for (let row = 0; row < currentNumRows; row++) {
        for (let col = 0; col < currentFeatureDim; col++) {
            const inputCell = document.getElementById(`inputTensor_${row}_${col}`);
            const outputCell = document.getElementById(`outputTensor_${row}_${col}`);
            if (inputCell) inputCell.className = 'tensor-cell';
            if (outputCell) outputCell.className = 'tensor-cell';
        }
    }
}

function updateCalculationsTable() {
    const tableBody = document.getElementById('calculationsTableBody');
    if (!tableBody) return;

    tableBody.innerHTML = '';

    for (let row = 0; row < currentNumRows; row++) {
        const tr = document.createElement('tr');
        tr.id = `calc-row-${row}`;
        tr.innerHTML = `
            <td>Row ${row + 1}</td>
            <td id="max-${row}">-</td>
            <td id="sum-exp-${row}">-</td>
            <td id="prob-sum-${row}">-</td>
            <td id="status-${row}">Pending</td>
        `;
        tableBody.appendChild(tr);
    }
}

function updateCalculationRow(rowIdx, max, sumExp, probSum, status) {
    const maxCell = document.getElementById(`max-${rowIdx}`);
    const sumExpCell = document.getElementById(`sum-exp-${rowIdx}`);
    const probSumCell = document.getElementById(`prob-sum-${rowIdx}`);
    const statusCell = document.getElementById(`status-${rowIdx}`);
    const rowElement = document.getElementById(`calc-row-${rowIdx}`);

    if (maxCell) maxCell.textContent = max !== undefined ? max.toFixed(3) : '-';
    if (sumExpCell) sumExpCell.textContent = sumExp !== undefined ? sumExp.toFixed(4) : '-';
    if (probSumCell) probSumCell.textContent = probSum !== undefined ? probSum.toFixed(6) : '-';
    if (statusCell) statusCell.textContent = status || 'Pending';

    // Update row styling based on status
    if (rowElement) {
        rowElement.className = '';
        if (status === 'Finding Max') {
            rowElement.className = 'current-processing';
        } else if (status === 'Computing Sum') {
            rowElement.className = 'current-processing';
        } else if (status === 'Computing Softmax') {
            rowElement.className = 'current-processing';
        } else if (status === 'Complete') {
            rowElement.className = 'completed';
        }
    }
}

function updateBlockSizeDisplay() {
    const blockSizeElement = document.getElementById('currentBlockSize');
    if (blockSizeElement) blockSizeElement.textContent = currentBlockSize;
}

function updateBlockSizeOptions() {
    const blockSizeSelect = document.getElementById('blockSize');
    const currentValue = blockSizeSelect.value;

    // Clear existing options
    blockSizeSelect.innerHTML = '';

    // Generate powers of 2 options up to current feature dimension
    for (let i = 1; i <= currentFeatureDim; i *= 2) {
        const option = document.createElement('option');
        option.value = i;
        option.textContent = i;
        blockSizeSelect.appendChild(option);
    }

    // Try to preserve the current selection, or default to a reasonable value
    let newValue = parseInt(currentValue) || 2;

    // Ensure the new value is a power of 2 and <= feature dimension
    if (newValue > currentFeatureDim || (newValue & (newValue - 1)) !== 0) {
        // Find the largest power of 2 that is <= feature dimension and >= 2
        if (currentFeatureDim >= 4) {
            newValue = 4;
        } else if (currentFeatureDim >= 2) {
            newValue = 2;
        } else {
            newValue = 1;
        }
    }

    blockSizeSelect.value = newValue;
}

function updateRowStats(rowIdx) {
    const row = outputData[rowIdx];
    const nonZeroValues = row.filter(val => val > 0);

    if (nonZeroValues.length === 0) {
        document.getElementById('maxProb').textContent = '-';
        document.getElementById('minProb').textContent = '-';
        document.getElementById('rowSum').textContent = '-';
        document.getElementById('entropy').textContent = '-';
        return;
    }

    const maxProb = Math.max(...nonZeroValues);
    const minProb = Math.min(...nonZeroValues);
    const rowSum = nonZeroValues.reduce((sum, val) => sum + val, 0);

    // Calculate entropy: -Σ(p * log2(p))
    let entropy = 0;
    for (let val of nonZeroValues) {
        if (val > 0) {
            entropy -= val * Math.log2(val);
        }
    }

    document.getElementById('maxProb').textContent = maxProb.toFixed(4);
    document.getElementById('minProb').textContent = minProb.toFixed(4);
    document.getElementById('rowSum').textContent = rowSum.toFixed(6);
    document.getElementById('entropy').textContent = entropy.toFixed(3);
}

async function runVisualization() {
    if (isRunning) return;
    isRunning = true;

    clearHighlights();
    document.getElementById('maxProb').textContent = '-';
    document.getElementById('minProb').textContent = '-';
    document.getElementById('rowSum').textContent = '-';
    document.getElementById('entropy').textContent = '-';

    // Process each row independently
    for (let rowIdx = 0; rowIdx < currentNumRows; rowIdx++) {
        document.getElementById('currentRow').textContent = `${rowIdx + 1}`;
        highlightRow(rowIdx, 'current-row');

        // Phase 1: Find Maximum
        document.getElementById('currentPhase').textContent = 'Finding Max';
        updateCalculationRow(rowIdx, undefined, undefined, undefined, 'Finding Max');

        let maxVal = -Infinity;

        for (let blockOffset = 0; blockOffset < currentFeatureDim; blockOffset += currentBlockSize) {
            const blockEnd = Math.min(blockOffset + currentBlockSize, currentFeatureDim);

            // Update UI
            document.getElementById('blockRange').textContent = `[${blockOffset}:${blockEnd - 1}]`;

            // Highlight current block
            highlightBlock(rowIdx, blockOffset, blockEnd, 'processing');

            // Find max in this block
            for (let i = blockOffset; i < blockEnd; i++) {
                maxVal = Math.max(maxVal, inputData[rowIdx][i]);
            }

            updateCalculationRow(rowIdx, maxVal, undefined, undefined, 'Finding Max');

            await sleep(400);

            // Mark block as processed
            highlightBlock(rowIdx, blockOffset, blockEnd, 'processed');
        }

        maxValues[rowIdx] = maxVal;

        // Phase 2: Compute Sum of Exponentials
        document.getElementById('currentPhase').textContent = 'Computing Sum';
        updateCalculationRow(rowIdx, maxVal, undefined, undefined, 'Computing Sum');

        let sumExp = 0;

        for (let blockOffset = 0; blockOffset < currentFeatureDim; blockOffset += currentBlockSize) {
            const blockEnd = Math.min(blockOffset + currentBlockSize, currentFeatureDim);

            // Update UI
            document.getElementById('blockRange').textContent = `[${blockOffset}:${blockEnd - 1}]`;

            // Highlight current block
            highlightBlock(rowIdx, blockOffset, blockEnd, 'processing');

            // Compute sum of exponentials in this block
            for (let i = blockOffset; i < blockEnd; i++) {
                const stableVal = inputData[rowIdx][i] - maxVal;
                sumExp += Math.exp(stableVal);
            }

            updateCalculationRow(rowIdx, maxVal, sumExp, undefined, 'Computing Sum');

            await sleep(400);

            // Mark block as processed
            highlightBlock(rowIdx, blockOffset, blockEnd, 'processed');
        }

        sumExpValues[rowIdx] = sumExp;

        // Phase 3: Compute Softmax
        document.getElementById('currentPhase').textContent = 'Computing Softmax';
        updateCalculationRow(rowIdx, maxVal, sumExp, undefined, 'Computing Softmax');

        // Pre-calculate all outputs for this row
        for (let i = 0; i < currentFeatureDim; i++) {
            const stableVal = inputData[rowIdx][i] - maxVal;
            const numerator = Math.exp(stableVal);
            outputData[rowIdx][i] = numerator / sumExp;
        }

        // Now update the UI in blocks for visualization
        for (let blockOffset = 0; blockOffset < currentFeatureDim; blockOffset += currentBlockSize) {
            const blockEnd = Math.min(blockOffset + currentBlockSize, currentFeatureDim);

            // Update UI
            document.getElementById('blockRange').textContent = `[${blockOffset}:${blockEnd - 1}]`;

            // Highlight current block
            highlightBlock(rowIdx, blockOffset, blockEnd, 'processing');

            // Update output display for this block
            for (let i = blockOffset; i < blockEnd; i++) {
                const outputCell = document.getElementById(`outputTensor_${rowIdx}_${i}`);
                if (outputCell) {
                    outputCell.textContent = outputData[rowIdx][i].toFixed(3);
                    outputCell.className = 'tensor-cell processing';
                }
            }

            await sleep(400);

            // Mark as processed
            for (let i = blockOffset; i < blockEnd; i++) {
                const outputCell = document.getElementById(`outputTensor_${rowIdx}_${i}`);
                if (outputCell) {
                    outputCell.className = 'tensor-cell processed';
                }
            }
        }

        // Calculate probability sum for verification
        const probSum = outputData[rowIdx].reduce((sum, val) => sum + val, 0);

        // Mark row as completely processed
        updateCalculationRow(rowIdx, maxVal, sumExp, probSum, 'Complete');

        // Keep input row highlighted as processed (green)
        highlightRow(rowIdx, 'processed');

        // Update stats for this row
        updateRowStats(rowIdx);

        await sleep(300);
    }

    // Final state
    document.getElementById('currentRow').textContent = 'Complete';
    document.getElementById('blockRange').textContent = 'All processed';
    document.getElementById('currentPhase').textContent = 'Complete';

    isRunning = false;
}

function reset() {
    clearHighlights();
    document.getElementById('currentRow').textContent = '-';
    document.getElementById('blockRange').textContent = '-';
    document.getElementById('currentPhase').textContent = '-';
    document.getElementById('maxProb').textContent = '-';
    document.getElementById('minProb').textContent = '-';
    document.getElementById('rowSum').textContent = '-';
    document.getElementById('entropy').textContent = '-';

    // Reset output data
    for (let row = 0; row < currentNumRows; row++) {
        outputData[row].fill(0);
    }
    updateTensorDisplay('outputTensor', outputData, false);

    // Reset calculations table
    updateCalculationsTable();
}

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

// Event listeners - these will be attached when the DOM loads
document.addEventListener('DOMContentLoaded', function () {
    // Event listeners
    document.getElementById('numRows').addEventListener('change', initializeData);
    document.getElementById('featureDim').addEventListener('change', initializeData);
    document.getElementById('blockSize').addEventListener('change', initializeData);

    // Initialize on load
    initializeData();
});