class MoERoutingVisualizer {
    constructor(containerId) {
        this.container = document.getElementById(containerId);
        this.isAnimating = false;
        this.animationSpeed = 1500;
        this.numTokens = 4;
        this.numExperts = 4;
        this.expertsPerToken = 2;

        this.init();
    }

    init() {
        this.container.innerHTML = `
            <style>
                .moe-container {
                    font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
                    max-width: 900px;
                    margin: 0 auto;
                    padding: 20px;
                    background: #1a1a1a;
                    border-radius: 12px;
                    color: #e0e0e0;
                }
                .moe-controls {
                    display: flex;
                    gap: 15px;
                    margin-bottom: 25px;
                    flex-wrap: wrap;
                    align-items: center;
                }
                .moe-control-group {
                    display: flex;
                    flex-direction: column;
                    gap: 5px;
                }
                .moe-control-group label {
                    font-size: 12px;
                    color: #999;
                    text-transform: uppercase;
                    letter-spacing: 0.5px;
                }
                .moe-control-group input {
                    padding: 8px 12px;
                    border: 1px solid #333;
                    border-radius: 6px;
                    background: #2a2a2a;
                    color: #e0e0e0;
                    width: 80px;
                }
                .moe-btn {
                    padding: 10px 20px;
                    border: none;
                    border-radius: 6px;
                    background: #4CAF50;
                    color: white;
                    cursor: pointer;
                    font-weight: 500;
                    transition: background 0.3s;
                    margin-top: auto;
                }
                .moe-btn:hover {
                    background: #45a049;
                }
                .moe-btn:disabled {
                    background: #333;
                    cursor: not-allowed;
                }
                .moe-visualization {
                    position: relative;
                    min-height: 600px;
                    background: #252525;
                    border-radius: 8px;
                    padding: 30px;
                }
                .moe-layer {
                    display: flex;
                    justify-content: space-around;
                    align-items: center;
                    margin-bottom: 50px;
                    position: relative;
                }
                .moe-token {
                    background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                    padding: 15px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    font-size: 14px;
                    font-weight: 600;
                    min-width: 80px;
                    text-align: center;
                    transition: transform 0.3s;
                }
                .moe-token:hover {
                    transform: scale(1.05);
                }
                .moe-expert {
                    background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
                    padding: 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    font-size: 13px;
                    font-weight: 600;
                    min-width: 100px;
                    text-align: center;
                    transition: transform 0.3s, background 0.3s;
                    position: relative;
                }
                .moe-expert.mlp2 {
                    background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                }
                .moe-expert:hover {
                    transform: scale(1.05);
                }
                .moe-expert-label {
                    font-size: 11px;
                    opacity: 0.9;
                    margin-top: 5px;
                }
                .moe-expert-count {
                    position: absolute;
                    top: -10px;
                    right: -10px;
                    background: #ff4444;
                    color: white;
                    border-radius: 50%;
                    width: 24px;
                    height: 24px;
                    display: flex;
                    align-items: center;
                    justify-content: center;
                    font-size: 11px;
                    font-weight: bold;
                    box-shadow: 0 2px 4px rgba(0,0,0,0.4);
                }
                .moe-output {
                    background: linear-gradient(135deg, #fa709a 0%, #fee140 100%);
                    padding: 15px 20px;
                    border-radius: 8px;
                    box-shadow: 0 4px 6px rgba(0,0,0,0.3);
                    font-size: 14px;
                    font-weight: 600;
                    min-width: 80px;
                    text-align: center;
                }
                .moe-section-title {
                    text-align: center;
                    font-size: 16px;
                    font-weight: 600;
                    margin-bottom: 20px;
                    color: #4CAF50;
                    text-transform: uppercase;
                    letter-spacing: 1px;
                }
                .moe-routing-info {
                    background: #2a2a2a;
                    padding: 15px;
                    border-radius: 8px;
                    margin-top: 20px;
                    font-size: 13px;
                    line-height: 1.6;
                }
                .moe-routing-info h4 {
                    margin: 0 0 10px 0;
                    color: #4CAF50;
                }
                .moe-routing-info code {
                    background: #1a1a1a;
                    padding: 2px 6px;
                    border-radius: 3px;
                    color: #64B5F6;
                }
                svg {
                    position: absolute;
                    top: 0;
                    left: 0;
                    width: 100%;
                    height: 100%;
                    pointer-events: none;
                    z-index: 0;
                }
                .moe-layer > * {
                    position: relative;
                    z-index: 1;
                }
                .routing-line {
                    stroke-width: 2;
                    fill: none;
                    opacity: 0;
                    transition: opacity 0.3s;
                }
                .routing-line.active {
                    opacity: 0.6;
                }
            </style>
            <div class="moe-container">
                <div class="moe-controls">
                    <div class="moe-control-group">
                        <label>Tokens</label>
                        <input type="number" id="numTokens" min="1" max="8" value="${this.numTokens}">
                    </div>
                    <div class="moe-control-group">
                        <label>Experts</label>
                        <input type="number" id="numExperts" min="2" max="8" value="${this.numExperts}">
                    </div>
                    <div class="moe-control-group">
                        <label>Experts/Token</label>
                        <input type="number" id="expertsPerToken" min="1" max="4" value="${this.expertsPerToken}">
                    </div>
                    <button class="moe-btn" id="animateBtn">Animate Routing</button>
                </div>
                <div class="moe-visualization" id="visualization">
                    <!-- Visualization will be rendered here -->
                </div>
                <div class="moe-routing-info" id="routingInfo">
                    <h4>MoE Routing Explanation</h4>
                    <p>Click "Animate Routing" to see how tokens are routed through the expert network.</p>
                    <p><strong>MLP Block 1 (Pink):</strong> First transformation layer with SwiGLU activation</p>
                    <p><strong>MLP Block 2 (Blue):</strong> Second transformation layer that produces final outputs</p>
                </div>
            </div>
        `;

        this.setupEventListeners();
        this.renderVisualization();
    }

    setupEventListeners() {
        document.getElementById('numTokens').addEventListener('change', (e) => {
            this.numTokens = parseInt(e.target.value);
            this.renderVisualization();
        });

        document.getElementById('numExperts').addEventListener('change', (e) => {
            this.numExperts = parseInt(e.target.value);
            this.renderVisualization();
        });

        document.getElementById('expertsPerToken').addEventListener('change', (e) => {
            this.expertsPerToken = Math.min(parseInt(e.target.value), this.numExperts);
            this.renderVisualization();
        });

        document.getElementById('animateBtn').addEventListener('click', () => {
            this.animateRouting();
        });
    }

    generateRouting() {
        const routing = [];
        for (let i = 0; i < this.numTokens; i++) {
            const experts = new Set();
            while (experts.size < this.expertsPerToken) {
                experts.add(Math.floor(Math.random() * this.numExperts));
            }
            routing.push({
                token: i,
                experts: Array.from(experts).sort((a, b) => a - b),
                weights: Array.from(experts).map(() => Math.random()).map(w => w / Array.from(experts).reduce((sum, _) => sum + Math.random(), 0))
            });
        }
        return routing;
    }

    renderVisualization() {
        const viz = document.getElementById('visualization');
        const routing = this.generateRouting();

        // Count tokens per expert for both blocks
        const mlp1Counts = new Array(this.numExperts).fill(0);
        const mlp2Counts = new Array(this.numExperts).fill(0);
        routing.forEach(r => {
            r.experts.forEach(e => {
                mlp1Counts[e]++;
                mlp2Counts[e]++;
            });
        });

        viz.innerHTML = `
            <svg id="routingSvg"></svg>

            <div class="moe-section-title">Input Tokens</div>
            <div class="moe-layer" id="tokenLayer">
                ${Array.from({length: this.numTokens}, (_, i) => `
                    <div class="moe-token" data-token="${i}">Token ${i}</div>
                `).join('')}
            </div>

            <div class="moe-section-title">MLP Block 1 (Experts)</div>
            <div class="moe-layer" id="mlp1Layer">
                ${Array.from({length: this.numExperts}, (_, i) => `
                    <div class="moe-expert" data-expert="${i}" data-block="1">
                        <div>Expert ${i}</div>
                        <div class="moe-expert-label">MLP1</div>
                        <div class="moe-expert-count">${mlp1Counts[i]}</div>
                    </div>
                `).join('')}
            </div>

            <div class="moe-section-title">MLP Block 2 (Experts)</div>
            <div class="moe-layer" id="mlp2Layer">
                ${Array.from({length: this.numExperts}, (_, i) => `
                    <div class="moe-expert mlp2" data-expert="${i}" data-block="2">
                        <div>Expert ${i}</div>
                        <div class="moe-expert-label">MLP2</div>
                        <div class="moe-expert-count">${mlp2Counts[i]}</div>
                    </div>
                `).join('')}
            </div>

            <div class="moe-section-title">Output Tokens</div>
            <div class="moe-layer" id="outputLayer">
                ${Array.from({length: this.numTokens}, (_, i) => `
                    <div class="moe-output" data-output="${i}">Out ${i}</div>
                `).join('')}
            </div>
        `;

        this.routing = routing;
        this.updateRoutingInfo();
    }

    updateRoutingInfo() {
        const info = document.getElementById('routingInfo');
        const routingDetails = this.routing.map(r =>
            `<div><code>Token ${r.token}</code> → Experts [${r.experts.join(', ')}]</div>`
        ).join('');

        info.innerHTML = `
            <h4>Current Routing Configuration</h4>
            ${routingDetails}
            <p style="margin-top: 15px;"><strong>Process:</strong></p>
            <ol style="margin: 5px 0; padding-left: 20px;">
                <li>Each token is processed by top-${this.expertsPerToken} experts in MLP Block 1</li>
                <li>SwiGLU activation is applied</li>
                <li>Same experts process the token in MLP Block 2</li>
                <li>Expert outputs are weighted and combined</li>
            </ol>
        `;
    }

    async animateRouting() {
        if (this.isAnimating) return;
        this.isAnimating = true;

        const btn = document.getElementById('animateBtn');
        btn.disabled = true;
        btn.textContent = 'Animating...';

        const svg = document.getElementById('routingSvg');
        svg.innerHTML = '';

        // Animate each token sequentially
        for (let i = 0; i < this.routing.length; i++) {
            await this.animateToken(this.routing[i], svg);
            await this.sleep(500);
        }

        await this.sleep(1000);
        svg.innerHTML = '';

        btn.disabled = false;
        btn.textContent = 'Animate Routing';
        this.isAnimating = false;
    }

    async animateToken(routingInfo, svg) {
        const tokenEl = document.querySelector(`[data-token="${routingInfo.token}"]`);

        // Animate Token -> MLP1 experts
        for (const expertIdx of routingInfo.experts) {
            const mlp1Expert = document.querySelector(`[data-expert="${expertIdx}"][data-block="1"]`);
            const line1 = this.drawLine(tokenEl, mlp1Expert, '#667eea', svg);
            line1.classList.add('active');
            await this.sleep(300);
        }

        await this.sleep(500);

        // Animate MLP1 -> MLP2 for same experts
        for (const expertIdx of routingInfo.experts) {
            const mlp1Expert = document.querySelector(`[data-expert="${expertIdx}"][data-block="1"]`);
            const mlp2Expert = document.querySelector(`[data-expert="${expertIdx}"][data-block="2"]`);
            const line2 = this.drawLine(mlp1Expert, mlp2Expert, '#f093fb', svg);
            line2.classList.add('active');
            await this.sleep(300);
        }

        await this.sleep(500);

        // Animate MLP2 -> Output
        const outputEl = document.querySelector(`[data-output="${routingInfo.token}"]`);
        for (const expertIdx of routingInfo.experts) {
            const mlp2Expert = document.querySelector(`[data-expert="${expertIdx}"][data-block="2"]`);
            const line3 = this.drawLine(mlp2Expert, outputEl, '#4facfe', svg);
            line3.classList.add('active');
            await this.sleep(300);
        }
    }

    drawLine(fromEl, toEl, color, svg) {
        const from = fromEl.getBoundingClientRect();
        const to = toEl.getBoundingClientRect();
        const svgRect = svg.getBoundingClientRect();

        const x1 = from.left + from.width / 2 - svgRect.left;
        const y1 = from.top + from.height / 2 - svgRect.top;
        const x2 = to.left + to.width / 2 - svgRect.left;
        const y2 = to.top + to.height / 2 - svgRect.top;

        const line = document.createElementNS('http://www.w3.org/2000/svg', 'path');
        const midY = (y1 + y2) / 2;
        const d = `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2}`;

        line.setAttribute('d', d);
        line.setAttribute('class', 'routing-line');
        line.setAttribute('stroke', color);
        svg.appendChild(line);

        return line;
    }

    sleep(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize when DOM is ready
if (typeof window !== 'undefined') {
    window.MoERoutingVisualizer = MoERoutingVisualizer;
}