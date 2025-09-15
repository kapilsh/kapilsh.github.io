import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, BarChart, Bar, ScatterChart, Scatter, ReferenceLine } from 'recharts';

const MXFP4Visualization = () => {
    const [selectedView, setSelectedView] = useState('mapping');

    // MXFP4 value mapping (16 possible values)
    const mxfp4Values = [
        { code: '0000', binary: 0, value: 0.0, sign: 0, exp: 0, mantissa: 0 },
        { code: '0001', binary: 1, value: 0.5, sign: 0, exp: 0, mantissa: 1 },
        { code: '0010', binary: 2, value: 1.0, sign: 0, exp: 1, mantissa: 0 },
        { code: '0011', binary: 3, value: 1.5, sign: 0, exp: 1, mantissa: 1 },
        { code: '0100', binary: 4, value: 2.0, sign: 0, exp: 2, mantissa: 0 },
        { code: '0101', binary: 5, value: 3.0, sign: 0, exp: 2, mantissa: 1 },
        { code: '0110', binary: 6, value: 4.0, sign: 0, exp: 3, mantissa: 0 },
        { code: '0111', binary: 7, value: 6.0, sign: 0, exp: 3, mantissa: 1 },
        { code: '1000', binary: 8, value: -0.0, sign: 1, exp: 0, mantissa: 0 },
        { code: '1001', binary: 9, value: -0.5, sign: 1, exp: 0, mantissa: 1 },
        { code: '1010', binary: 10, value: -1.0, sign: 1, exp: 1, mantissa: 0 },
        { code: '1011', binary: 11, value: -1.5, sign: 1, exp: 1, mantissa: 1 },
        { code: '1100', binary: 12, value: -2.0, sign: 1, exp: 2, mantissa: 0 },
        { code: '1101', binary: 13, value: -3.0, sign: 1, exp: 2, mantissa: 1 },
        { code: '1110', binary: 14, value: -4.0, sign: 1, exp: 3, mantissa: 0 },
        { code: '1111', binary: 15, value: -6.0, sign: 1, exp: 3, mantissa: 1 }
    ];

    // Function to find nearest MXFP4 value for quantization
    const quantizeToMXFP4 = (value) => {
        let minDiff = Infinity;
        let nearestValue = null;

        mxfp4Values.forEach(mxfp4 => {
            const diff = Math.abs(value - mxfp4.value);
            if (diff < minDiff) {
                minDiff = diff;
                nearestValue = mxfp4;
            }
        });

        return nearestValue;
    };

    // Generate mapping data for visualization
    const generateMappingData = () => {
        const data = [];
        for (let i = -7; i <= 7; i += 0.1) {
            const original = parseFloat(i.toFixed(1));
            const quantized = quantizeToMXFP4(original);
            data.push({
                original: original,
                quantized: quantized.value,
                error: original - quantized.value,
                code: quantized.code
            });
        }
        return data;
    };

    // Generate distribution data (simulate neural network weights)
    const generateDistributionData = () => {
        const samples = 10000;
        const distribution = {};

        // Initialize counts for each MXFP4 value
        mxfp4Values.forEach(val => {
            distribution[val.value] = 0;
        });

        // Generate normally distributed values (typical for NN weights)
        for (let i = 0; i < samples; i++) {
            // Box-Muller transform for normal distribution
            const u1 = Math.random();
            const u2 = Math.random();
            const normal = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);

            // Scale to fit roughly in MXFP4 range
            const scaledValue = normal * 1.5;

            // Clamp to MXFP4 range
            const clampedValue = Math.max(-6, Math.min(6, scaledValue));

            // Quantize and count
            const quantized = quantizeToMXFP4(clampedValue);
            distribution[quantized.value]++;
        }

        // Convert to chart format
        return mxfp4Values.map(val => ({
            value: val.value,
            code: val.code,
            count: distribution[val.value],
            probability: distribution[val.value] / samples
        })).sort((a, b) => a.value - b.value);
    };

    const mappingData = generateMappingData();
    const distributionData = generateDistributionData();

    // Create quantization boundaries data
    const quantizationBoundaries = () => {
        const boundaries = [];
        const sortedValues = [...mxfp4Values].sort((a, b) => a.value - b.value);

        for (let i = 0; i < sortedValues.length - 1; i++) {
            const current = sortedValues[i].value;
            const next = sortedValues[i + 1].value;
            const boundary = (current + next) / 2;
            boundaries.push({
                x: boundary,
                label: `${current} | ${next}`
            });
        }
        return boundaries;
    };

    const boundaries = quantizationBoundaries();

    return (
        <div className="w-full max-w-6xl mx-auto p-6 bg-white">
            <div className="mb-6">
                <h2 className="text-2xl font-bold mb-4">MXFP4 Value Mapping and Distribution</h2>

                <div className="flex gap-4 mb-6">
                    <button
                        onClick={() => setSelectedView('mapping')}
                        className={`px-4 py-2 rounded ${selectedView === 'mapping' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                    >
                        Value Mapping
                    </button>
                    <button
                        onClick={() => setSelectedView('distribution')}
                        className={`px-4 py-2 rounded ${selectedView === 'distribution' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                    >
                        Probability Distribution
                    </button>
                    <button
                        onClick={() => setSelectedView('table')}
                        className={`px-4 py-2 rounded ${selectedView === 'table' ? 'bg-blue-500 text-white' : 'bg-gray-200'}`}
                    >
                        Value Table
                    </button>
                </div>
            </div>

            {selectedView === 'mapping' && (
                <div className="space-y-6">
                    <div>
                        <h3 className="text-lg font-semibold mb-2">Continuous to Discrete Mapping</h3>
                        <p className="text-sm text-gray-600 mb-4">
                            Shows how continuous input values (blue line) map to discrete MXFP4 values (red dots).
                            The step function illustrates the quantization process.
                        </p>
                        <ResponsiveContainer width="100%" height={400}>
                            <LineChart data={mappingData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    dataKey="original"
                                    domain={[-7, 7]}
                                    type="number"
                                    label={{ value: 'Original Value', position: 'insideBottom', offset: -5 }}
                                />
                                <YAxis
                                    domain={[-7, 7]}
                                    label={{ value: 'Quantized Value', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip
                                    formatter={(value, name) => [
                                        typeof value === 'number' ? value.toFixed(1) : value,
                                        name === 'original' ? 'Original' : name === 'quantized' ? 'MXFP4' : 'Error'
                                    ]}
                                />
                                <Legend />
                                <Line
                                    type="monotone"
                                    dataKey="original"
                                    stroke="#3b82f6"
                                    strokeWidth={1}
                                    dot={false}
                                    name="y = x (identity)"
                                />
                                <Line
                                    type="stepAfter"
                                    dataKey="quantized"
                                    stroke="#ef4444"
                                    strokeWidth={2}
                                    dot={false}
                                    name="MXFP4 Quantized"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>

                    <div>
                        <h3 className="text-lg font-semibold mb-2">Quantization Error</h3>
                        <p className="text-sm text-gray-600 mb-4">
                            Shows the error (difference) between original and quantized values.
                            Notice larger errors in sparse regions of the MXFP4 value space.
                        </p>
                        <ResponsiveContainer width="100%" height={300}>
                            <LineChart data={mappingData}>
                                <CartesianGrid strokeDasharray="3 3" />
                                <XAxis
                                    dataKey="original"
                                    domain={[-7, 7]}
                                    type="number"
                                    label={{ value: 'Original Value', position: 'insideBottom', offset: -5 }}
                                />
                                <YAxis
                                    label={{ value: 'Quantization Error', angle: -90, position: 'insideLeft' }}
                                />
                                <Tooltip
                                    formatter={(value, name) => [value.toFixed(3), 'Error']}
                                />
                                <ReferenceLine y={0} stroke="#666" strokeDasharray="2 2" />
                                <Line
                                    type="monotone"
                                    dataKey="error"
                                    stroke="#f59e0b"
                                    strokeWidth={1}
                                    dot={false}
                                    name="Quantization Error"
                                />
                            </LineChart>
                        </ResponsiveContainer>
                    </div>
                </div>
            )}

            {selectedView === 'distribution' && (
                <div>
                    <h3 className="text-lg font-semibold mb-2">Value Distribution (Simulated Neural Network Weights)</h3>
                    <p className="text-sm text-gray-600 mb-4">
                        Shows how normally distributed values (typical of neural network weights)
                        map to the 16 discrete MXFP4 values. Values cluster around zero due to the normal distribution.
                    </p>
                    <ResponsiveContainer width="100%" height={530}>
                        <BarChart data={distributionData}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis
                                dataKey="value"
                                label={{ value: 'MXFP4 Value', position: 'insideBottom', offset: -5 }}
                            />
                            <YAxis
                                label={{ value: 'Probability', angle: -90, position: 'insideLeft' }}
                            />
                            <Tooltip
                                formatter={(value, name) => [
                                    name === 'probability' ? (value * 100).toFixed(1) + '%' : value,
                                    name === 'probability' ? 'Probability' : 'Count'
                                ]}
                                labelFormatter={(value) => `MXFP4 Value: ${value}`}
                            />
                            <Legend />
                            <Bar
                                dataKey="probability"
                                fill="#3b82f6"
                                name="Probability"
                            />
                        </BarChart>
                    </ResponsiveContainer>

                    <div className="mt-4 text-sm text-gray-600">
                        <p><strong>Key Observations:</strong></p>
                        <ul className="list-disc list-inside space-y-1">
                            <li>Values near zero (0.0, ±0.5, ±1.0) have highest probability</li>
                            <li>Extreme values (±4.0, ±6.0) are rarely used</li>
                            <li>The distribution reflects typical neural network weight patterns</li>
                            <li>Non-uniform spacing in MXFP4 values affects representation efficiency</li>
                        </ul>
                    </div>
                </div>
            )}

            {selectedView === 'table' && (
                <div>
                    <h3 className="text-lg font-semibold mb-4">Complete MXFP4 Value Table</h3>
                    <div className="overflow-x-auto">
                        <table className="w-full border-collapse border border-gray-300">
                            <thead>
                                <tr className="bg-gray-100">
                                    <th className="border border-gray-300 px-4 py-2">Binary Code</th>
                                    <th className="border border-gray-300 px-4 py-2">Sign</th>
                                    <th className="border border-gray-300 px-4 py-2">Exponent</th>
                                    <th className="border border-gray-300 px-4 py-2">Mantissa</th>
                                    <th className="border border-gray-300 px-4 py-2">Value</th>
                                    <th className="border border-gray-300 px-4 py-2">Decimal Index</th>
                                </tr>
                            </thead>
                            <tbody>
                                {mxfp4Values.map((item, index) => (
                                    <tr key={index} className={index % 2 === 0 ? 'bg-gray-50' : 'bg-white'}>
                                        <td className="border border-gray-300 px-4 py-2 font-mono">{item.code}</td>
                                        <td className="border border-gray-300 px-4 py-2 text-center">{item.sign}</td>
                                        <td className="border border-gray-300 px-4 py-2 text-center">{item.exp.toString(2).padStart(2, '0')}</td>
                                        <td className="border border-gray-300 px-4 py-2 text-center">{item.mantissa}</td>
                                        <td className="border border-gray-300 px-4 py-2 text-center font-semibold">{item.value}</td>
                                        <td className="border border-gray-300 px-4 py-2 text-center">{item.binary}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>

                    <div className="mt-6 grid grid-cols-1 md:grid-cols-2 gap-6">
                        <div>
                            <h4 className="font-semibold mb-2">Value Statistics</h4>
                            <div className="text-sm space-y-1">
                                <p><strong>Total values:</strong> 16</p>
                                <p><strong>Range:</strong> [-6.0, 6.0]</p>
                                <p><strong>Zero values:</strong> 2 (0.0, -0.0)</p>
                                <p><strong>Positive values:</strong> 7</p>
                                <p><strong>Negative values:</strong> 7</p>
                            </div>
                        </div>

                        <div>
                            <h4 className="font-semibold mb-2">Spacing Analysis</h4>
                            <div className="text-sm space-y-1">
                                <p><strong>Smallest gap:</strong> 0.5 (between 0.0 and 0.5)</p>
                                <p><strong>Largest gap:</strong> 2.0 (between 4.0 and 6.0)</p>
                                <p><strong>Non-uniform spacing:</strong> Logarithmic-like distribution</p>
                                <p><strong>Dense region:</strong> [-1.5, 1.5] (8 out of 16 values)</p>
                            </div>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
};

export default MXFP4Visualization;