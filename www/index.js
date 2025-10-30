import init, {
    JsTensor,
    JsModel, 
    JsDataset,
    AsyncTrainer,
    Utils,
    DataConverter
} from '../pkg/multilayer_perceptron.js';

class MLPDemo {
    constructor() {
        this.model = null;
        this.trainer = null;
        this.dataset = null;
        this.isTraining = false;
        this.trainingHistory = [];
        this.charts = {};
        
        this.initializeEventListeners();
        this.initializeCharts();
    }

    async initialize() {
        try {
            await init();
            this.log('WebAssembly module loaded successfully!', 'success');
            this.log('Ready for neural network training', 'info');
        } catch (error) {
            this.log(`Failed to load WebAssembly: ${error}`, 'error');
        }
    }

    initializeEventListeners() {
        document.getElementById('start-training').addEventListener('click', () => this.startTraining());
        document.getElementById('stop-training').addEventListener('click', () => this.stopTraining());
        document.getElementById('generate-data').addEventListener('click', () => this.generateData());
        document.getElementById('predict').addEventListener('click', () => this.makePrediction());
    }

    initializeCharts() {
        // Initialize canvas contexts
        this.dataPlotCtx = document.getElementById('data-plot').getContext('2d');
        this.lossChartCtx = document.getElementById('loss-chart').getContext('2d');
        this.decisionBoundaryCtx = document.getElementById('decision-boundary').getContext('2d');
    }

    generateData() {
        try {
            // Generate synthetic 2D binary classification data
            const numSamples = 200;
            const features = [];
            const labels = [];

            for (let i = 0; i < numSamples; i++) {
                const x = (Math.random() - 0.5) * 4; // Range: -2 to 2
                const y = (Math.random() - 0.5) * 4; // Range: -2 to 2
                
                // Simple circular decision boundary
                const label = (x * x + y * y) < 1.5 ? 1.0 : 0.0;
                
                features.push(new Float64Array([x, y]));
                labels.push(label);
            }

            // Convert to JsDataset
            const featuresArray = new Array(features.length);
            for (let i = 0; i < features.length; i++) {
                featuresArray[i] = features[i];
            }
            
            this.dataset = new JsDataset(featuresArray, new Float64Array(labels));
            
            this.log(`Generated ${numSamples} samples for binary classification`, 'success');
            document.getElementById('data-info').textContent = 
                `Dataset: ${numSamples} samples, 2 features, circular decision boundary`;
            
            this.plotData();
        } catch (error) {
            this.log(`Failed to generate data: ${error}`, 'error');
        }
    }

    plotData() {
        if (!this.dataset) return;

        const canvas = document.getElementById('data-plot');
        const ctx = this.dataPlotCtx;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        try {
            const features = this.dataset.features_tensor();
            const labels = this.dataset.labels_tensor();
            
            const featuresData = features.data();
            const labelsData = labels.data();
            
            const numSamples = this.dataset.len();
            
            // Plot data points
            for (let i = 0; i < numSamples; i++) {
                const x = (featuresData[i * 2] + 2) * canvas.width / 4; // Scale to canvas
                const y = (2 - featuresData[i * 2 + 1]) * canvas.height / 4; // Scale and flip Y
                const label = labelsData[i];
                
                ctx.beginPath();
                ctx.arc(x, y, 3, 0, 2 * Math.PI);
                ctx.fillStyle = label > 0.5 ? '#48bb78' : '#f56565';
                ctx.fill();
                ctx.strokeStyle = '#2d3748';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
            
            // Draw axes
            ctx.beginPath();
            ctx.moveTo(canvas.width / 2, 0);
            ctx.lineTo(canvas.width / 2, canvas.height);
            ctx.moveTo(0, canvas.height / 2);
            ctx.lineTo(canvas.width, canvas.height / 2);
            ctx.strokeStyle = '#cbd5e0';
            ctx.lineWidth = 1;
            ctx.stroke();
            
        } catch (error) {
            this.log(`Failed to plot data: ${error}`, 'error');
        }
    }

    async startTraining() {
        if (!this.dataset) {
            this.log('Please generate data first!', 'warning');
            return;
        }

        if (this.isTraining) return;

        try {
            this.isTraining = true;
            document.getElementById('start-training').disabled = true;
            document.getElementById('stop-training').disabled = false;

            const learningRate = parseFloat(document.getElementById('learning-rate').value);
            const epochs = parseInt(document.getElementById('epochs').value);
            const hiddenSize = parseInt(document.getElementById('hidden-size').value);

            // Create model
            this.model = Utils.create_binary_classifier(2, hiddenSize);
            
            this.log(`Created model: Input(2) -> Hidden(${hiddenSize}) -> Output(1)`, 'info');
            document.getElementById('model-summary').innerHTML = `
                <strong>Model Architecture:</strong><br>
                Input Layer: 2 neurons<br>
                Hidden Layer: ${hiddenSize} neurons (ReLU)<br>
                Output Layer: 1 neuron (Sigmoid)<br>
                <strong>Total Parameters:</strong> ${this.model.param_count()}
            `;

            // Create trainer
            this.trainer = new AsyncTrainer(learningRate, epochs, 32);
            
            this.log(`Starting training: LR=${learningRate}, Epochs=${epochs}`, 'info');
            document.getElementById('total-epochs').textContent = epochs;
            
            this.trainingHistory = [];

            // Start async training with progress callback
            const history = await this.trainer.train_async(
                this.model, 
                this.dataset, 
                (progress) => this.onTrainingProgress(progress)
            );

            if (this.isTraining) {
                this.log('Training completed successfully!', 'success');
                this.plotDecisionBoundary();
            }

        } catch (error) {
            this.log(`Training failed: ${error}`, 'error');
        } finally {
            this.isTraining = false;
            document.getElementById('start-training').disabled = false;
            document.getElementById('stop-training').disabled = true;
        }
    }

    stopTraining() {
        if (this.isTraining) {
            this.isTraining = false;
            this.log('Training stopped by user', 'warning');
        }
    }

    onTrainingProgress(progress) {
        if (!this.isTraining) return;

        document.getElementById('current-epoch').textContent = progress.epoch;
        document.getElementById('current-loss').textContent = progress.loss.toFixed(4);
        document.getElementById('current-accuracy').textContent = (progress.accuracy * 100).toFixed(1) + '%';
        
        const progressPercent = (progress.epoch / parseInt(document.getElementById('epochs').value)) * 100;
        document.getElementById('progress-fill').style.width = progressPercent + '%';

        this.trainingHistory.push({
            epoch: progress.epoch,
            loss: progress.loss,
            accuracy: progress.accuracy
        });

        this.updateLossChart();

        this.log(`Epoch ${progress.epoch}: Loss=${progress.loss.toFixed(4)}, Accuracy=${(progress.accuracy * 100).toFixed(1)}%`, 'info');
    }

    updateLossChart() {
        const canvas = document.getElementById('loss-chart');
        const ctx = this.lossChartCtx;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        if (this.trainingHistory.length < 2) return;

        const maxEpochs = this.trainingHistory[this.trainingHistory.length - 1].epoch;
        const maxLoss = Math.max(...this.trainingHistory.map(h => h.loss));
        
        // Draw loss curve
        ctx.beginPath();
        ctx.strokeStyle = '#667eea';
        ctx.lineWidth = 2;
        
        this.trainingHistory.forEach((point, index) => {
            const x = (point.epoch / maxEpochs) * canvas.width;
            const y = canvas.height - (point.loss / maxLoss) * canvas.height;
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();

        // Draw axes
        ctx.beginPath();
        ctx.strokeStyle = '#cbd5e0';
        ctx.lineWidth = 1;
        ctx.moveTo(0, canvas.height);
        ctx.lineTo(canvas.width, canvas.height);
        ctx.moveTo(0, 0);
        ctx.lineTo(0, canvas.height);
        ctx.stroke();
    }

    plotDecisionBoundary() {
        if (!this.model) return;

        const canvas = document.getElementById('decision-boundary');
        const ctx = this.decisionBoundaryCtx;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);

        try {
            const resolution = 50;
            
            // Create a grid of points to evaluate
            for (let i = 0; i < resolution; i++) {
                for (let j = 0; j < resolution; j++) {
                    const x = (i / (resolution - 1)) * 4 - 2; // Range: -2 to 2
                    const y = (j / (resolution - 1)) * 4 - 2; // Range: -2 to 2
                    
                    // Create input tensor
                    const input = JsTensor.new(new Float64Array([x, y]), 1, 2);
                    
                    // Make prediction
                    const output = this.model.forward(input);
                    const prediction = output.data()[0];
                    
                    // Color based on prediction
                    const canvasX = (x + 2) * canvas.width / 4;
                    const canvasY = (2 - y) * canvas.height / 4;
                    
                    const intensity = Math.floor(prediction * 255);
                    ctx.fillStyle = `rgba(${255 - intensity}, ${intensity}, 100, 0.3)`;
                    ctx.fillRect(canvasX - 2, canvasY - 2, 4, 4);
                }
            }
            
            // Overlay the original data points
            if (this.dataset) {
                const features = this.dataset.features_tensor();
                const labels = this.dataset.labels_tensor();
                
                const featuresData = features.data();
                const labelsData = labels.data();
                const numSamples = this.dataset.len();
                
                for (let i = 0; i < numSamples; i++) {
                    const x = (featuresData[i * 2] + 2) * canvas.width / 4;
                    const y = (2 - featuresData[i * 2 + 1]) * canvas.height / 4;
                    const label = labelsData[i];
                    
                    ctx.beginPath();
                    ctx.arc(x, y, 3, 0, 2 * Math.PI);
                    ctx.fillStyle = label > 0.5 ? '#48bb78' : '#f56565';
                    ctx.fill();
                    ctx.strokeStyle = '#2d3748';
                    ctx.lineWidth = 2;
                    ctx.stroke();
                }
            }
            
        } catch (error) {
            this.log(`Failed to plot decision boundary: ${error}`, 'error');
        }
    }

    makePrediction() {
        if (!this.model) {
            this.log('Please train a model first!', 'warning');
            return;
        }

        try {
            const x = parseFloat(document.getElementById('input-x').value);
            const y = parseFloat(document.getElementById('input-y').value);
            
            const input = JsTensor.new(new Float64Array([x, y]), 1, 2);
            const output = this.model.forward(input);
            const prediction = output.data()[0];
            
            const confidence = Math.abs(prediction - 0.5) * 2; // 0 to 1
            const predictedClass = prediction > 0.5 ? 'Class A' : 'Class B';
            
            document.getElementById('prediction-result').innerHTML = `
                <strong>Input:</strong> (${x.toFixed(2)}, ${y.toFixed(2)})<br>
                <strong>Prediction:</strong> ${predictedClass}<br>
                <strong>Probability:</strong> ${(prediction * 100).toFixed(1)}%<br>
                <strong>Confidence:</strong> ${(confidence * 100).toFixed(1)}%
            `;
            
            this.log(`Prediction for (${x.toFixed(2)}, ${y.toFixed(2)}): ${predictedClass} (${(prediction * 100).toFixed(1)}%)`, 'info');
            
        } catch (error) {
            this.log(`Prediction failed: ${error}`, 'error');
        }
    }

    log(message, type = 'info') {
        const logOutput = document.getElementById('log-output');
        const timestamp = new Date().toLocaleTimeString();
        const logEntry = document.createElement('div');
        logEntry.className = `log-entry ${type}`;
        logEntry.textContent = `[${timestamp}] ${message}`;
        
        logOutput.appendChild(logEntry);
        logOutput.scrollTop = logOutput.scrollHeight;
    }
}

// Initialize the demo when the page loads
document.addEventListener('DOMContentLoaded', async () => {
    const demo = new MLPDemo();
    await demo.initialize();
    
    // Generate initial data
    demo.generateData();
});