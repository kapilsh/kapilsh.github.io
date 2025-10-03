/**
 * Triton GPU IR Visualizer
 * Parses Triton IR and creates an interactive step-by-step visualization
 */

class TritonIRVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.instructions = [];
        this.dependencies = new Map();
        this.currentStep = 0;
        this.animationSpeed = 1000;
        this.visibleNodes = new Set(); // Track which nodes are currently visible
        this.selectedNode = null; // Currently selected node for subgraph view
        this.viewMode = 'all'; // 'all' or 'subgraph'
    }

    /**
     * Parse Triton IR text into structured instructions
     */
    parseIR(irText) {
        this.instructions = [];
        this.dependencies = new Map();

        const lines = irText.split('\n');
        let inFunction = false;
        let instructionIndex = 0;

        for (let line of lines) {
            line = line.trim();

            // Skip empty lines and metadata
            if (!line || line.startsWith('#') || line.startsWith('module') ||
                line.startsWith('}') || line.startsWith('loc(')) {
                continue;
            }

            // Detect function start
            if (line.includes('tt.func')) {
                inFunction = true;
                continue;
            }

            if (!inFunction) continue;

            // Parse SSA instructions
            const instruction = this.parseInstruction(line, instructionIndex);
            if (instruction) {
                this.instructions.push(instruction);
                this.dependencies.set(instruction.result, instruction);
                instructionIndex++;
            }
        }

        this.buildDependencyGraph();
        return this.instructions;
    }

    /**
     * Parse a single instruction line
     */
    parseInstruction(line, index) {
        // Match SSA result pattern: %0 = ...
        const resultMatch = line.match(/^(%\w+|%cst[_\d]*)\s*=\s*(.+?)(?:\s+loc\(|$)/);

        if (!resultMatch) {
            // Handle instructions without result (tt.store, tt.return)
            if (line.includes('tt.store') || line.includes('tt.return')) {
                const deps = this.extractDependencies(line);
                return {
                    index,
                    result: `inst_${index}`,
                    operation: line.split(/\s+/)[0],
                    fullText: line,
                    dependencies: deps,
                    type: this.getOperationType(line),
                    shape: this.extractShape(line)
                };
            }
            return null;
        }

        const result = resultMatch[1];
        const rest = resultMatch[2];
        const operation = rest.split(/\s+/)[0];
        const deps = this.extractDependencies(rest);

        return {
            index,
            result,
            operation,
            fullText: line,
            dependencies: deps,
            type: this.getOperationType(rest),
            shape: this.extractShape(rest)
        };
    }

    /**
     * Extract dependencies (other SSA values used)
     */
    extractDependencies(text) {
        const deps = [];
        const regex = /%\w+|%cst[_\d]*/g;
        let match;

        while ((match = regex.exec(text)) !== null) {
            deps.push(match[0]);
        }

        return deps;
    }

    /**
     * Determine operation type for coloring
     */
    getOperationType(text) {
        if (text.includes('arith.constant')) return 'constant';
        if (text.includes('tt.get_program_id')) return 'input';
        if (text.includes('tt.make_range')) return 'range';
        if (text.includes('tt.load')) return 'load';
        if (text.includes('tt.store')) return 'store';
        if (text.includes('arith.muli') || text.includes('arith.mulf')) return 'arithmetic';
        if (text.includes('arith.addi') || text.includes('arith.addf')) return 'arithmetic';
        if (text.includes('tt.addptr')) return 'pointer';
        if (text.includes('tt.splat')) return 'broadcast';
        if (text.includes('tt.reduce')) return 'reduce';
        if (text.includes('tt.reshape')) return 'reshape';
        if (text.includes('arith.cmpi')) return 'compare';
        return 'other';
    }

    /**
     * Extract tensor shape information
     */
    extractShape(text) {
        const tensorMatch = text.match(/tensor<([^>]+)>/);
        if (tensorMatch) {
            return tensorMatch[1];
        }
        return null;
    }

    /**
     * Build dependency graph with levels
     */
    buildDependencyGraph() {
        // Topological sort to determine levels
        const levels = new Map();
        const visited = new Set();

        const computeLevel = (inst) => {
            if (visited.has(inst.result)) {
                return levels.get(inst.result);
            }

            visited.add(inst.result);
            let maxDepLevel = -1;

            for (const dep of inst.dependencies) {
                const depInst = this.dependencies.get(dep);
                if (depInst) {
                    maxDepLevel = Math.max(maxDepLevel, computeLevel(depInst));
                }
            }

            const level = maxDepLevel + 1;
            levels.set(inst.result, level);
            inst.level = level;
            return level;
        };

        this.instructions.forEach(inst => computeLevel(inst));
    }

    /**
     * Get all dependencies (ancestors) of a node recursively
     */
    getDependencies(nodeResult, visited = new Set()) {
        if (visited.has(nodeResult)) return visited;

        const inst = this.dependencies.get(nodeResult);
        if (!inst) return visited;

        visited.add(nodeResult);

        inst.dependencies.forEach(dep => {
            this.getDependencies(dep, visited);
        });

        return visited;
    }

    /**
     * Get all dependents (descendants) of a node recursively
     */
    getDependents(nodeResult, visited = new Set()) {
        if (visited.has(nodeResult)) return visited;

        visited.add(nodeResult);

        this.instructions.forEach(inst => {
            if (inst.dependencies.includes(nodeResult)) {
                this.getDependents(inst.result, visited);
            }
        });

        return visited;
    }

    /**
     * Get subgraph containing a node and its dependencies/dependents
     */
    getSubgraph(nodeResult, includeDependents = true) {
        const subgraph = new Set();

        // Add the node itself
        subgraph.add(nodeResult);

        // Add all dependencies (ancestors)
        const deps = this.getDependencies(nodeResult);
        deps.forEach(dep => subgraph.add(dep));

        // Add all dependents (descendants) if requested
        if (includeDependents) {
            const dependents = this.getDependents(nodeResult);
            dependents.forEach(dep => subgraph.add(dep));
        }

        return subgraph;
    }

    /**
     * Set which nodes should be visible
     */
    setVisibleNodes(nodeResults) {
        this.visibleNodes = new Set(nodeResults);
    }

    /**
     * Show all nodes
     */
    showAllNodes() {
        this.viewMode = 'all';
        this.selectedNode = null;
        this.visibleNodes.clear();
        this.updateVisualization();
    }

    /**
     * Focus on a specific node and its subgraph
     */
    focusOnNode(nodeResult) {
        this.viewMode = 'subgraph';
        this.selectedNode = nodeResult;
        const subgraph = this.getSubgraph(nodeResult, true);
        this.setVisibleNodes(subgraph);
        this.updateVisualization();
    }

    /**
     * Render the visualization
     */
    render() {
        this.container.innerHTML = '';

        // Create SVG canvas
        const svg = this.createSVG();
        this.container.appendChild(svg);

        // Draw the graph after a short delay to ensure SVG is rendered
        setTimeout(() => this.drawGraph(svg), 10);
    }

    /**
     * Create control panel
     */
    createControls() {
        const controls = document.createElement('div');
        controls.className = 'ir-controls';
        controls.style.cssText = 'margin: 20px 0; padding: 15px; background: #f5f5f5; border-radius: 8px;';

        const stepInfo = document.createElement('div');
        stepInfo.className = 'step-info';
        stepInfo.style.cssText = 'margin-bottom: 10px; font-weight: bold;';
        stepInfo.textContent = `Step: ${this.currentStep} / ${this.instructions.length}`;
        controls.appendChild(stepInfo);

        // Step controls
        const buttonContainer = document.createElement('div');
        buttonContainer.style.cssText = 'display: flex; gap: 10px; align-items: center; margin-bottom: 15px;';

        const prevBtn = this.createButton('Previous', () => this.previousStep());
        const nextBtn = this.createButton('Next', () => this.nextStep());
        const resetBtn = this.createButton('Reset', () => this.reset());
        const playBtn = this.createButton('Play All', () => this.playAll());

        const speedLabel = document.createElement('label');
        speedLabel.textContent = 'Speed: ';
        speedLabel.style.marginLeft = '20px';

        const speedInput = document.createElement('input');
        speedInput.type = 'range';
        speedInput.min = '100';
        speedInput.max = '2000';
        speedInput.value = '1000';
        speedInput.step = '100';
        speedInput.addEventListener('input', (e) => {
            this.animationSpeed = 2100 - parseInt(e.target.value);
        });

        buttonContainer.appendChild(prevBtn);
        buttonContainer.appendChild(nextBtn);
        buttonContainer.appendChild(resetBtn);
        buttonContainer.appendChild(playBtn);
        buttonContainer.appendChild(speedLabel);
        buttonContainer.appendChild(speedInput);

        controls.appendChild(buttonContainer);

        // Subgraph controls
        const subgraphContainer = document.createElement('div');
        subgraphContainer.style.cssText = 'display: flex; gap: 10px; align-items: center; padding-top: 15px; border-top: 1px solid #ddd;';

        const viewLabel = document.createElement('label');
        viewLabel.textContent = 'Focus on node: ';
        viewLabel.style.fontWeight = 'bold';

        const nodeSelect = document.createElement('select');
        nodeSelect.style.cssText = 'padding: 5px 10px; border-radius: 4px; border: 1px solid #ccc;';

        const allOption = document.createElement('option');
        allOption.value = 'all';
        allOption.textContent = 'Show All Nodes';
        nodeSelect.appendChild(allOption);

        this.instructions.forEach(inst => {
            const option = document.createElement('option');
            option.value = inst.result;
            option.textContent = `${inst.result} (${inst.operation})`;
            nodeSelect.appendChild(option);
        });

        nodeSelect.addEventListener('change', (e) => {
            if (e.target.value === 'all') {
                this.showAllNodes();
            } else {
                this.focusOnNode(e.target.value);
            }
        });

        const showDepsBtn = this.createButton('Show Only Dependencies', () => {
            if (this.selectedNode) {
                const deps = this.getDependencies(this.selectedNode);
                deps.add(this.selectedNode);
                this.setVisibleNodes(deps);
                this.viewMode = 'subgraph';
                this.updateVisualization();
            }
        });
        showDepsBtn.style.fontSize = '12px';

        subgraphContainer.appendChild(viewLabel);
        subgraphContainer.appendChild(nodeSelect);
        subgraphContainer.appendChild(showDepsBtn);

        controls.appendChild(subgraphContainer);

        this.stepInfoElement = stepInfo;
        this.nodeSelectElement = nodeSelect;
        return controls;
    }

    /**
     * Create a button element
     */
    createButton(text, onClick) {
        const btn = document.createElement('button');
        btn.textContent = text;
        btn.onclick = onClick;
        btn.style.cssText = 'padding: 8px 16px; cursor: pointer; border: none; background: #007bff; color: white; border-radius: 4px;';
        btn.onmouseover = () => btn.style.background = '#0056b3';
        btn.onmouseout = () => btn.style.background = '#007bff';
        return btn;
    }

    /**
     * Create SVG canvas
     */
    createSVG() {
        const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
        svg.setAttribute('width', '100%');
        svg.setAttribute('height', '600');
        svg.style.cssText = 'background: white; display: block;';
        return svg;
    }

    /**
     * Draw the instruction graph
     */
    drawGraph(svg) {
        // Clear existing content
        while (svg.firstChild) {
            svg.removeChild(svg.firstChild);
        }

        const nodeWidth = 180;
        const nodeHeight = 50;
        const levelHeight = 90;
        const horizontalSpacing = 30;

        // Filter instructions based on view mode
        const visibleInstructions = this.viewMode === 'all'
            ? this.instructions
            : this.instructions.filter(inst => this.visibleNodes.has(inst.result));

        // Group instructions by level
        const levelGroups = new Map();
        let maxLevel = 0;
        let maxNodesInLevel = 0;
        visibleInstructions.forEach(inst => {
            const level = inst.level || 0;
            maxLevel = Math.max(maxLevel, level);
            if (!levelGroups.has(level)) {
                levelGroups.set(level, []);
            }
            levelGroups.get(level).push(inst);
            maxNodesInLevel = Math.max(maxNodesInLevel, levelGroups.get(level).length);
        });

        // Calculate dynamic dimensions based on graph size
        const width = Math.max(1000, maxNodesInLevel * (nodeWidth + horizontalSpacing) + 80);
        const height = Math.max(500, (maxLevel + 1) * levelHeight + 100);

        svg.setAttribute('width', width);
        svg.setAttribute('height', height);
        svg.setAttribute('viewBox', `0 0 ${width} ${height}`);

        // Calculate positions
        const positions = new Map();
        levelGroups.forEach((insts, level) => {
            const levelWidth = insts.length * (nodeWidth + horizontalSpacing);
            const startX = Math.max(40, (width - levelWidth) / 2);

            insts.forEach((inst, i) => {
                positions.set(inst.result, {
                    x: startX + i * (nodeWidth + horizontalSpacing),
                    y: level * levelHeight + 50
                });
            });
        });

        // Add arrowhead marker definition
        this.addArrowheadMarker(svg);

        // Draw edges first (so they appear behind nodes)
        const edgesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        edgesGroup.setAttribute('class', 'edges');
        svg.appendChild(edgesGroup);

        visibleInstructions.forEach((inst, idx) => {
            if (idx >= this.currentStep) return;

            const toPos = positions.get(inst.result);
            if (!toPos) return;

            inst.dependencies.forEach(dep => {
                const fromPos = positions.get(dep);
                if (!fromPos) return;

                // Skip if dependency is not visible
                if (this.viewMode === 'subgraph' && !this.visibleNodes.has(dep)) {
                    return;
                }

                const line = this.createEdge(
                    fromPos.x + nodeWidth / 2,
                    fromPos.y + nodeHeight,
                    toPos.x + nodeWidth / 2,
                    toPos.y,
                    idx < this.currentStep
                );
                edgesGroup.appendChild(line);
            });
        });

        // Add level labels
        const labelsGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        labelsGroup.setAttribute('class', 'level-labels');
        svg.appendChild(labelsGroup);

        levelGroups.forEach((insts, level) => {
            const label = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            label.setAttribute('x', 10);
            label.setAttribute('y', level * levelHeight + 50 + nodeHeight / 2);
            label.setAttribute('font-size', '11');
            label.setAttribute('font-weight', 'bold');
            label.setAttribute('fill', '#999');
            label.textContent = `L${level}`;
            labelsGroup.appendChild(label);
        });

        // Draw nodes
        const nodesGroup = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        nodesGroup.setAttribute('class', 'nodes');
        svg.appendChild(nodesGroup);

        this.instructions.forEach((inst, idx) => {
            const pos = positions.get(inst.result);
            if (!pos) return;

            const isActive = idx < this.currentStep;
            const isCurrent = idx === this.currentStep - 1;
            const node = this.createNode(inst, pos.x, pos.y, nodeWidth, nodeHeight, isActive, isCurrent);
            nodesGroup.appendChild(node);
        });
    }

    /**
     * Add arrowhead marker definition to SVG
     */
    addArrowheadMarker(svg) {
        if (svg.querySelector('#arrowhead')) return;

        const defs = document.createElementNS('http://www.w3.org/2000/svg', 'defs');
        const marker = document.createElementNS('http://www.w3.org/2000/svg', 'marker');
        marker.setAttribute('id', 'arrowhead');
        marker.setAttribute('markerWidth', '10');
        marker.setAttribute('markerHeight', '10');
        marker.setAttribute('refX', '9');
        marker.setAttribute('refY', '3');
        marker.setAttribute('orient', 'auto');
        const polygon = document.createElementNS('http://www.w3.org/2000/svg', 'polygon');
        polygon.setAttribute('points', '0 0, 10 3, 0 6');
        polygon.setAttribute('fill', '#666');
        marker.appendChild(polygon);
        defs.appendChild(marker);
        svg.insertBefore(defs, svg.firstChild);
    }

    /**
     * Create an edge (arrow) between nodes
     */
    createEdge(x1, y1, x2, y2, isActive) {
        const path = document.createElementNS('http://www.w3.org/2000/svg', 'path');

        // Use elbow connector (orthogonal lines) with offset to avoid overlapping nodes
        // Add a small vertical offset from the node before going horizontal
        const offset = 15; // pixels to go down before turning
        const midY = (y1 + y2) / 2;

        // Create elbow path with offset: down a bit, across, then down to target
        const d = `M ${x1} ${y1} L ${x1} ${y1 + offset} L ${x1} ${midY} L ${x2} ${midY} L ${x2} ${y2 - offset} L ${x2} ${y2}`;

        path.setAttribute('d', d);
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', isActive ? '#555' : '#e0e0e0');
        path.setAttribute('stroke-width', isActive ? '1.5' : '1');
        path.setAttribute('marker-end', 'url(#arrowhead)');
        path.setAttribute('opacity', isActive ? '0.6' : '0.2');

        return path;
    }

    /**
     * Create a node representing an instruction
     */
    createNode(inst, x, y, width, height, isActive, isCurrent) {
        const group = document.createElementNS('http://www.w3.org/2000/svg', 'g');
        group.setAttribute('class', 'node');
        group.style.cursor = 'pointer';

        // Make node clickable to focus on it
        group.addEventListener('click', () => {
            this.focusOnNode(inst.result);
            if (this.nodeSelectElement) {
                this.nodeSelectElement.value = inst.result;
            }
        });

        // Node rectangle
        const rect = document.createElementNS('http://www.w3.org/2000/svg', 'rect');
        rect.setAttribute('x', x);
        rect.setAttribute('y', y);
        rect.setAttribute('width', width);
        rect.setAttribute('height', height);
        rect.setAttribute('rx', '5');

        const color = this.getNodeColor(inst.type);
        rect.setAttribute('fill', isActive ? color : '#f0f0f0');
        rect.setAttribute('stroke', isCurrent ? '#ff0000' : '#333');
        rect.setAttribute('stroke-width', isCurrent ? '3' : '1');
        rect.setAttribute('opacity', isActive ? '1' : '0.3');

        group.appendChild(rect);

        // Result text
        const resultText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        resultText.setAttribute('x', x + width / 2);
        resultText.setAttribute('y', y + 16);
        resultText.setAttribute('text-anchor', 'middle');
        resultText.setAttribute('font-size', '12');
        resultText.setAttribute('font-weight', 'bold');
        resultText.setAttribute('fill', isActive ? '#000' : '#999');
        resultText.textContent = inst.result;
        group.appendChild(resultText);

        // Operation text
        const opText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
        opText.setAttribute('x', x + width / 2);
        opText.setAttribute('y', y + 32);
        opText.setAttribute('text-anchor', 'middle');
        opText.setAttribute('font-size', '10');
        opText.setAttribute('fill', isActive ? '#333' : '#999');
        opText.textContent = inst.operation;
        group.appendChild(opText);

        // Shape info (if available) - only show for specific types
        if (inst.shape && inst.shape.length < 30) {
            const shapeText = document.createElementNS('http://www.w3.org/2000/svg', 'text');
            shapeText.setAttribute('x', x + width / 2);
            shapeText.setAttribute('y', y + 44);
            shapeText.setAttribute('text-anchor', 'middle');
            shapeText.setAttribute('font-size', '8');
            shapeText.setAttribute('fill', isActive ? '#666' : '#aaa');
            shapeText.textContent = inst.shape.length > 20 ? inst.shape.substring(0, 20) + '...' : inst.shape;
            group.appendChild(shapeText);
        }

        // Tooltip on hover
        const title = document.createElementNS('http://www.w3.org/2000/svg', 'title');
        title.textContent = inst.fullText;
        group.appendChild(title);

        return group;
    }

    /**
     * Get color for node based on operation type
     */
    getNodeColor(type) {
        const colors = {
            'constant': '#e3f2fd',
            'input': '#c8e6c9',
            'range': '#fff9c4',
            'load': '#ffccbc',
            'store': '#f8bbd0',
            'arithmetic': '#b3e5fc',
            'pointer': '#dcedc8',
            'broadcast': '#ffe0b2',
            'reduce': '#f48fb1',
            'reshape': '#ce93d8',
            'compare': '#ffab91',
            'other': '#e0e0e0'
        };
        return colors[type] || colors['other'];
    }

    /**
     * Step forward
     */
    nextStep() {
        if (this.currentStep < this.instructions.length) {
            this.currentStep++;
            this.updateVisualization();
        }
    }

    /**
     * Step backward
     */
    previousStep() {
        if (this.currentStep > 0) {
            this.currentStep--;
            this.updateVisualization();
        }
    }

    /**
     * Reset to beginning
     */
    reset() {
        this.currentStep = 0;
        this.updateVisualization();
    }

    /**
     * Play all steps automatically
     */
    async playAll() {
        this.reset();
        while (this.currentStep < this.instructions.length) {
            await new Promise(resolve => setTimeout(resolve, this.animationSpeed));
            this.nextStep();
        }
    }

    /**
     * Update the visualization
     */
    updateVisualization() {
        this.stepInfoElement.textContent = `Step: ${this.currentStep} / ${this.instructions.length}`;
        const svg = this.container.querySelector('svg');
        this.drawGraph(svg);
    }
}

// Initialize when DOM is ready
function initTritonIRVisualizer(vizContainerId, controlsContainerId, irText) {
    const visualizer = new TritonIRVisualizer(vizContainerId);
    visualizer.parseIR(irText);

    // Create controls in separate container
    const controlsContainer = document.getElementById(controlsContainerId);
    controlsContainer.innerHTML = '';
    const controls = visualizer.createControls();
    controlsContainer.appendChild(controls);

    // Render visualization
    visualizer.render();
    return visualizer;
}
