let inputData = [];
let weightData = [];
let outputData = [];
let currentNumRows = 2;
let currentFeatureDim = 8;
let currentBlockSize = 4;
let currentEpsilon = 1e-6;
let isRunning = false;
let rmsValues = [];
let rstdValues = [];

function initializeData() {
    const prevFeatureDim = currentFeatureDim;
    const prevNumRows = currentNumRows;

    currentNumRows = parseInt(document.getElementById('numRows').value);
    currentFeatureDim = parseInt(document.getElementById('featureDim').value);
    currentEpsilon = parseFloat(document.getElementById('epsilon').value);

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
    rmsValues = [];
    rstdValues = [];
    for (let row = 0; row < currentNumRows; row++) {
        outputData[row] = new Array(currentFeatureDim).fill(0);
        rmsValues[row] = 0;
        rstdValues[row] = 0;
    }

    // Only regenerate weights if feature dimension changed or weights don't exist
    if (!weightData || weightData.length !== currentFeatureDim) {
        weightData = [];
        for (let i = 0; i < currentFeatureDim; i++) {
            weightData[i] = 0.8 + Math.random() * 0.4;
        }
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
    const rowVariation = (rowIndex + 1) * 0.5;

    for (let i = 0; i < dim; i++) {
        if (rowIndex % 2 === 0) {
            preset[i] = rowVariation * (2.0 + Math.sin(i * 0.5)) * (i % 2 === 0 ? 1 : -0.6);
        } else {
            preset[i] = rowVariation * (1.5 + Math.cos(i * 0.8)) * (i % 3 === 0 ? -1 : 0.8);
        }
    }
    return preset;
}

function generateRandomInput() {
    if (isRunning) return;

    for (let row = 0; row < currentNumRows; row++) {
        for (let i = 0; i < currentFeatureDim; i++) {
            inputData[row][i] = (Math.random() - 0.5) * 4; // Random values between -2 and 2
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
        if (inputCell) {
            inputCell.className = `tensor-cell ${className}`;
        }
    }
}

function highlightBlock(rowIdx, startIdx, endIdx, className) {
    for (let i = startIdx; i < Math.min(endIdx, currentFeatureDim); i++) {
        const cell = document.getElementById(`inputTensor_${rowIdx}_${i}`);
        if (cell) {
            cell.className = `tensor-cell ${className}`;
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
            <td id="sum-squares-${row}">-</td>
            <td id="mean-square-${row}">-</td>
            <td id="rms-${row}">-</td>
            <td id="rstd-${row}">-</td>
            <td id="status-${row}">Pending</td>
        `;
        tableBody.appendChild(tr);
    }
}

function updateCalculationRow(rowIdx, sumOfSquares, meanSquare, rms, rstd, status) {
    const sumSquaresCell = document.getElementById(`sum-squares-${rowIdx}`);
    const meanSquareCell = document.getElementById(`mean-square-${rowIdx}`);
    const rmsCell = document.getElementById(`rms-${rowIdx}`);
    const rstdCell = document.getElementById(`rstd-${rowIdx}`);
    const statusCell = document.getElementById(`status-${rowIdx}`);
    const rowElement = document.getElementById(`calc-row-${rowIdx}`);

    if (sumSquaresCell) sumSquaresCell.textContent = sumOfSquares !== undefined ? sumOfSquares.toFixed(6) : '-';
    if (meanSquareCell) meanSquareCell.textContent = meanSquare !== undefined ? meanSquare.toFixed(6) : '-';
    if (rmsCell) rmsCell.textContent = rms !== undefined ? rms.toFixed(6) : '-';
    if (rstdCell) rstdCell.textContent = rstd !== undefined ? rstd.toFixed(6) : '-';
    if (statusCell) statusCell.textContent = status || 'Pending';

    // Update row styling based on status
    if (rowElement) {
        rowElement.className = '';
        if (status === 'Processing') {
            rowElement.className = 'current-processing';
        } else if (status === 'Complete') {
            rowElement.className = 'completed';
        }
    }
}

function updateBlockSizeDisplay() {
    const blockSizeElement = document.getElementById('currentBlockSize');
    const epsilonElement = document.getElementById('currentEpsilon');

    if (blockSizeElement) blockSizeElement.textContent = currentBlockSize;
    if (epsilonElement) epsilonElement.textContent = currentEpsilon;
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

async function runVisualization() {
    if (isRunning) return;
    isRunning = true;

    clearHighlights();
    document.getElementById('currentRMS').textContent = '-';
    document.getElementById('currentRstd').textContent = '-';


    // Process each row independently
    for (let rowIdx = 0; rowIdx < currentNumRows; rowIdx++) {
        document.getElementById('currentRow').textContent = `${rowIdx + 1}`;
        highlightRow(rowIdx, 'current-row');

        // Update calculations table to show current row being processed
        updateCalculationRow(rowIdx, undefined, undefined, undefined, undefined, 'Processing');

        // Step 1: Calculate sum of squares for this row using blocks
        let sumOfSquares = 0;

        for (let blockOffset = 0; blockOffset < currentFeatureDim; blockOffset += currentBlockSize) {
            const blockEnd = Math.min(blockOffset + currentBlockSize, currentFeatureDim);

            // Update UI
            document.getElementById('blockRange').textContent = `[${blockOffset}:${blockEnd - 1}]`;

            // Highlight current block
            highlightBlock(rowIdx, blockOffset, blockEnd, 'processing');

            // Calculate sum of squares for this block
            let blockSum = 0;
            for (let i = blockOffset; i < blockEnd; i++) {
                blockSum += inputData[rowIdx][i] * inputData[rowIdx][i];
            }
            sumOfSquares += blockSum;

            // Update calculations table with current sum of squares
            updateCalculationRow(rowIdx, sumOfSquares, undefined, undefined, undefined, 'Processing');

            await sleep(600);

            // Mark block as processed
            highlightBlock(rowIdx, blockOffset, blockEnd, 'processed');
        }

        // Step 2: Calculate mean square and rstd for this row
        const meanSquare = sumOfSquares / currentFeatureDim;
        const rms = Math.sqrt(meanSquare);
        const rstd = 1 / Math.sqrt(meanSquare + currentEpsilon);

        rmsValues[rowIdx] = rms;
        rstdValues[rowIdx] = rstd;

        // Clear row highlighting now that block processing is done
        highlightRow(rowIdx, 'processed');

        // Update calculations table with all computed values
        updateCalculationRow(rowIdx, sumOfSquares, meanSquare, rms, rstd, 'Computing Output');

        document.getElementById('currentRMS').textContent = rms.toFixed(6);
        document.getElementById('currentRstd').textContent = rstd.toFixed(6);

        await sleep(800);

        // Step 3: Apply normalization for this row (block processing for visualization only)
        // Calculate all outputs for this row first (to ensure consistency regardless of block size)
        const rowRstd = rstdValues[rowIdx];
        for (let i = 0; i < currentFeatureDim; i++) {
            // Standard RMS normalization formula: output = (input * rstd) * weight
            outputData[rowIdx][i] = inputData[rowIdx][i] * rowRstd * weightData[i];
        }

        // Now update the UI in blocks for visualization
        for (let blockOffset = 0; blockOffset < currentFeatureDim; blockOffset += currentBlockSize) {
            const blockEnd = Math.min(blockOffset + currentBlockSize, currentFeatureDim);

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

        // Mark row as completely processed
        updateCalculationRow(rowIdx, sumOfSquares, meanSquare, rms, rstd, 'Complete');

        // Keep input row highlighted as processed (green)
        highlightRow(rowIdx, 'processed');

        await sleep(300);
    }

    // Don't clear highlights - keep all rows showing as processed
    document.getElementById('currentRow').textContent = 'Complete';
    document.getElementById('blockRange').textContent = 'All processed';

    isRunning = false;
}

function reset() {
    clearHighlights();
    document.getElementById('currentRow').textContent = '-';
    document.getElementById('blockRange').textContent = '-';
    document.getElementById('currentRMS').textContent = '-';
    document.getElementById('currentRstd').textContent = '-';

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
    document.getElementById('epsilon').addEventListener('change', () => {
        currentEpsilon = parseFloat(document.getElementById('epsilon').value);
    });

    // Initialize on load
    initializeData();
});