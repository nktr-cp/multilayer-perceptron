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
        this.validationHistory = [];
        this.charts = {};
        this.startTime = null;
        this.bestAccuracy = 0;
        this.bestEpoch = 0;
        this.lastMetrics = {};
        this.elapsedTimer = null;
        
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
        document.getElementById('reset-charts').addEventListener('click', () => this.resetCharts());
    }

    initializeCharts() {
        // Initialize canvas contexts for manual drawing
        this.dataPlotCtx = document.getElementById('data-plot').getContext('2d');
        this.decisionBoundaryCtx = document.getElementById('decision-boundary').getContext('2d');
        
        // Initialize Chart.js charts
        this.createLossChart();
        this.createAccuracyChart();
    }

    createLossChart() {
        const ctx = document.getElementById('loss-chart').getContext('2d');
        this.charts.loss = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Loss',
                    data: [],
                    borderColor: '#f56565',
                    backgroundColor: 'rgba(245, 101, 101, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.3
                }, {
                    label: 'Validation Loss',
                    data: [],
                    borderColor: '#4299e1',
                    backgroundColor: 'rgba(66, 153, 225, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: false
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    createAccuracyChart() {
        const ctx = document.getElementById('accuracy-chart').getContext('2d');
        this.charts.accuracy = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [{
                    label: 'Training Accuracy',
                    data: [],
                    borderColor: '#48bb78',
                    backgroundColor: 'rgba(72, 187, 120, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.3
                }, {
                    label: 'Validation Accuracy',
                    data: [],
                    borderColor: '#805ad5',
                    backgroundColor: 'rgba(128, 90, 213, 0.1)',
                    borderWidth: 2,
                    fill: false,
                    pointRadius: 0,
                    pointHoverRadius: 5,
                    tension: 0.3
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    title: {
                        display: false
                    },
                    legend: {
                        display: true,
                        position: 'top'
                    }
                },
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoch'
                        },
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Accuracy'
                        },
                        min: 0,
                        max: 1,
                        grid: {
                            display: true,
                            color: 'rgba(0, 0, 0, 0.1)'
                        },
                        ticks: {
                            callback: function(value) {
                                return (value * 100).toFixed(0) + '%';
                            }
                        }
                    }
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                }
            }
        });
    }

    resetCharts() {
        if (!this.charts.loss || !this.charts.accuracy) {
            this.initializeCharts();
        }

        if (!this.charts.loss || !this.charts.accuracy) {
            this.log('Charts are not initialized yet. Skipping reset.', 'warning');
            return;
        }

        this.trainingHistory = [];
        this.validationHistory = [];
        this.bestAccuracy = 0;
        this.bestEpoch = 0;
        this.lastMetrics = {};
        
        // Reset Chart.js charts
        this.charts.loss.data.labels = [];
        this.charts.loss.data.datasets[0].data = [];
        this.charts.loss.data.datasets[1].data = [];
        this.charts.loss.update();
        
        this.charts.accuracy.data.labels = [];
        this.charts.accuracy.data.datasets[0].data = [];
        this.charts.accuracy.data.datasets[1].data = [];
        this.charts.accuracy.update();
        
        // Reset progress display
        document.getElementById('current-epoch').textContent = '0';
        document.getElementById('current-loss').textContent = '0.000';
        document.getElementById('current-accuracy').textContent = '0.0%';
        document.getElementById('current-val-loss').textContent = '0.000';
        document.getElementById('current-val-accuracy').textContent = '0.0%';
        document.getElementById('best-accuracy').textContent = '0.0%';
        document.getElementById('best-epoch').textContent = '0';
        document.getElementById('progress-fill').style.width = '0%';
        document.getElementById('progress-text').textContent = '0%';
        document.getElementById('elapsed-time').textContent = '00:00';
        document.getElementById('eta-time').textContent = '--:--';
        
        // Reset trend indicators
        ['loss-trend', 'accuracy-trend', 'val-loss-trend', 'val-accuracy-trend'].forEach(id => {
            const el = document.getElementById(id);
            el.textContent = '—';
            el.className = 'trend-indicator';
        });
        
        // Reset status
        this.updateTrainingStatus('stopped', 'Ready to train');
        
        this.log('Charts and metrics reset', 'info');
    }

    updateTrainingStatus(status, message) {
        const statusLight = document.getElementById('status-light');
        const statusText = document.getElementById('training-status-text');
        
        statusLight.className = `status-light ${status}`;
        statusText.textContent = message;
    }

    formatTime(seconds) {
        const mins = Math.floor(seconds / 60);
        const secs = Math.floor(seconds % 60);
        return `${mins.toString().padStart(2, '0')}:${secs.toString().padStart(2, '0')}`;
    }

    updateElapsedTime() {
        if (this.startTime) {
            const elapsed = (Date.now() - this.startTime) / 1000;
            document.getElementById('elapsed-time').textContent = this.formatTime(elapsed);
        }
    }

    calculateTrend(current, previous) {
        if (previous === undefined) return '—';
        
        const diff = current - previous;
        const threshold = Math.abs(previous) * 0.01; // 1% threshold
        
        if (Math.abs(diff) < threshold) return '➡️'; // stable
        return diff < 0 ? '⬇️' : '⬆️'; // down or up
    }

    updateTrendIndicator(elementId, current, previous, isHigherBetter = true) {
        const element = document.getElementById(elementId);
        const trend = this.calculateTrend(current, previous);
        
        element.textContent = trend;
        
        if (trend === '—' || trend === '➡️') {
            element.className = 'trend-indicator stable';
        } else {
            const isImproving = isHigherBetter ? (trend === '⬆️') : (trend === '⬇️');
            element.className = `trend-indicator ${isImproving ? 'up' : 'down'}`;
        }
    }

    generateData() {
        try {
            const datasetType = document.getElementById('dataset-type').value;
            const noiseLevel = parseFloat(document.getElementById('noise-level').value);
            const numSamplesPerClass = parseInt(document.getElementById('num-samples').value);
            
            const features = [];
            const labels = [];
            
            // Helper function for Box-Muller transform (normal distribution)
            const gaussianRandom = (mean = 0, std = 1) => {
                let u = 0, v = 0;
                while(u === 0) u = Math.random();
                while(v === 0) v = Math.random();
                const z = Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
                return z * std + mean;
            };
            
            let patternName, difficulty, description;
            
            switch (datasetType) {
                case 'diagonal':
                    this.generateDiagonalClusters(features, labels, numSamplesPerClass, noiseLevel, gaussianRandom);
                    patternName = 'Diagonal Clusters';
                    difficulty = 'Easy';
                    description = 'Linearly separable with diagonal boundary';
                    break;
                    
                case 'circular':
                    this.generateCircularBoundary(features, labels, numSamplesPerClass, noiseLevel);
                    patternName = 'Circular Boundary';
                    difficulty = 'Medium';
                    description = 'Circular decision boundary';
                    break;
                    
                case 'xor':
                    this.generateXORPattern(features, labels, numSamplesPerClass, noiseLevel, gaussianRandom);
                    patternName = 'XOR Pattern';
                    difficulty = 'Hard';
                    description = 'Non-linearly separable XOR pattern';
                    break;
                    
                case 'spiral':
                    this.generateSpiralPattern(features, labels, numSamplesPerClass, noiseLevel);
                    patternName = 'Spiral Pattern';
                    difficulty = 'Very Hard';
                    description = 'Interleaved spiral arms';
                    break;
                    
                case 'moons':
                    this.generateTwoMoons(features, labels, numSamplesPerClass, noiseLevel);
                    patternName = 'Two Moons';
                    difficulty = 'Medium';
                    description = 'Two interlocked crescents';
                    break;
            }

            // Shuffle the data
            const combined = features.map((f, i) => ({ features: f, label: labels[i] }));
            for (let i = combined.length - 1; i > 0; i--) {
                const j = Math.floor(Math.random() * (i + 1));
                [combined[i], combined[j]] = [combined[j], combined[i]];
            }
            
            const shuffledFeatures = combined.map(item => item.features);
            const shuffledLabels = combined.map(item => item.label);

            // Convert to JsDataset
            this.dataset = new JsDataset(shuffledFeatures, new Float64Array(shuffledLabels));
            
            const class0Count = shuffledLabels.filter(l => l === 0.0).length;
            const class1Count = shuffledLabels.filter(l => l === 1.0).length;
            const totalSamples = shuffledFeatures.length;
            
            this.log(`Generated ${totalSamples} samples: ${patternName} pattern`, 'success');
            document.getElementById('data-info').innerHTML = 
                `<strong>Dataset:</strong> ${totalSamples} samples, 2 features<br>
                 <strong>Classes:</strong> ${class0Count} red dots (class 0), ${class1Count} green dots (class 1)<br>
                 <strong>Pattern:</strong> ${patternName}<br>
                 <strong>Difficulty:</strong> ${difficulty} - ${description}<br>
                 <strong>Noise Level:</strong> ${noiseLevel}`;
            
            this.plotData();
        } catch (error) {
            this.log(`Failed to generate data: ${error}`, 'error');
        }
    }

    generateDiagonalClusters(features, labels, numSamplesPerClass, noiseLevel, gaussianRandom) {
        // Class 0: Two clusters in bottom-left and top-right
        for (let i = 0; i < numSamplesPerClass; i++) {
            let x, y;
            if (i < numSamplesPerClass / 2) {
                x = gaussianRandom(-1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
                y = gaussianRandom(-1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
            } else {
                x = gaussianRandom(1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
                y = gaussianRandom(1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
            }
            features.push(new Float64Array([x, y]));
            labels.push(0.0);
        }
        
        // Class 1: Two clusters in top-left and bottom-right
        for (let i = 0; i < numSamplesPerClass; i++) {
            let x, y;
            if (i < numSamplesPerClass / 2) {
                x = gaussianRandom(-1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
                y = gaussianRandom(1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
            } else {
                x = gaussianRandom(1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
                y = gaussianRandom(-1.2, 0.3) + (Math.random() - 0.5) * noiseLevel;
            }
            features.push(new Float64Array([x, y]));
            labels.push(1.0);
        }
    }

    generateCircularBoundary(features, labels, numSamplesPerClass, noiseLevel) {
        for (let i = 0; i < numSamplesPerClass * 2; i++) {
            const x = (Math.random() - 0.5) * 4;
            const y = (Math.random() - 0.5) * 4;
            const radius = Math.sqrt(x * x + y * y);
            const label = radius < 1.5 ? 1.0 : 0.0;
            
            const noisyX = x + (Math.random() - 0.5) * noiseLevel * 2;
            const noisyY = y + (Math.random() - 0.5) * noiseLevel * 2;
            
            features.push(new Float64Array([noisyX, noisyY]));
            labels.push(label);
        }
    }

    generateXORPattern(features, labels, numSamplesPerClass, noiseLevel, gaussianRandom) {
        // Generate four clusters for XOR pattern
        const clusterCenters = [
            [-1, -1, 0], [1, 1, 0],   // Class 0
            [-1, 1, 1], [1, -1, 1]    // Class 1
        ];
        
        const samplesPerCluster = numSamplesPerClass / 2;
        
        for (const [cx, cy, label] of clusterCenters) {
            for (let i = 0; i < samplesPerCluster; i++) {
                const x = gaussianRandom(cx, 0.3) + (Math.random() - 0.5) * noiseLevel;
                const y = gaussianRandom(cy, 0.3) + (Math.random() - 0.5) * noiseLevel;
                
                features.push(new Float64Array([x, y]));
                labels.push(label);
            }
        }
    }

    generateSpiralPattern(features, labels, numSamplesPerClass, noiseLevel) {
        for (let i = 0; i < numSamplesPerClass; i++) {
            for (let label = 0; label < 2; label++) {
                const t = (i / numSamplesPerClass) * 4 * Math.PI;
                const r = t / (4 * Math.PI);
                const angle = t + label * Math.PI;
                
                const x = r * Math.cos(angle) + (Math.random() - 0.5) * noiseLevel;
                const y = r * Math.sin(angle) + (Math.random() - 0.5) * noiseLevel;
                
                features.push(new Float64Array([x, y]));
                labels.push(label);
            }
        }
    }

    generateTwoMoons(features, labels, numSamplesPerClass, noiseLevel) {
        for (let i = 0; i < numSamplesPerClass; i++) {
            // First moon (class 0)
            const angle1 = Math.PI * i / numSamplesPerClass;
            const x1 = Math.cos(angle1) + (Math.random() - 0.5) * noiseLevel;
            const y1 = Math.sin(angle1) + (Math.random() - 0.5) * noiseLevel;
            
            features.push(new Float64Array([x1, y1]));
            labels.push(0.0);
            
            // Second moon (class 1)
            const angle2 = Math.PI * (1 - i / numSamplesPerClass);
            const x2 = 1 - Math.cos(angle2) + (Math.random() - 0.5) * noiseLevel;
            const y2 = 0.5 - Math.sin(angle2) + (Math.random() - 0.5) * noiseLevel;
            
            features.push(new Float64Array([x2, y2]));
            labels.push(1.0);
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

    getHyperparameters() {
        return {
            learningRate: parseFloat(document.getElementById('learning-rate').value),
            epochs: parseInt(document.getElementById('epochs').value),
            batchSize: parseInt(document.getElementById('batch-size').value),
            hiddenSize: parseInt(document.getElementById('hidden-size').value),
            validationSplit: parseFloat(document.getElementById('validation-split').value),
            momentum: parseFloat(document.getElementById('momentum').value)
        };
    }

    async startTraining() {
        if (!this.dataset) {
            this.log('Please generate data first!', 'warning');
            return;
        }

        if (this.isTraining) return;

        try {
            this.isTraining = true;
            this.startTime = Date.now();
            this.bestAccuracy = 0;
            this.bestEpoch = 0;
            this.lastMetrics = {};
            
            document.getElementById('start-training').disabled = true;
            document.getElementById('stop-training').disabled = false;
            
            this.updateTrainingStatus('training', 'Training in progress...');

            const params = this.getHyperparameters();
            
            this.log(`Hyperparameters: LR=${params.learningRate}, Epochs=${params.epochs}, Batch=${params.batchSize}, Hidden=${params.hiddenSize}`, 'info');
            
            // Create model
            this.model = Utils.create_binary_classifier(2, params.hiddenSize);
            
            this.log(`Created model: Input(2) -> Hidden(${params.hiddenSize}) -> Output(1)`, 'info');
            document.getElementById('model-summary').innerHTML = `
                <strong>Model Architecture:</strong><br>
                Input Layer: 2 neurons<br>
                Hidden Layer: ${params.hiddenSize} neurons (ReLU)<br>
                Output Layer: 1 neuron (Sigmoid)<br>
                <strong>Total Parameters:</strong> ${this.model.param_count()}<br>
                <strong>Hyperparameters:</strong><br>
                Learning Rate: ${params.learningRate}<br>
                Batch Size: ${params.batchSize}<br>
                Validation Split: ${(params.validationSplit * 100).toFixed(0)}%<br>
                Momentum: ${params.momentum}
            `;

            // Create trainer
            this.trainer = new AsyncTrainer(params.learningRate, params.epochs, params.batchSize);
            
            document.getElementById('total-epochs').textContent = params.epochs;
            
            // Reset history
            this.trainingHistory = [];
            this.validationHistory = [];
            
            // Start elapsed time timer
            this.elapsedTimer = setInterval(() => this.updateElapsedTime(), 1000);

            // Start async training with progress callback
            const history = await this.trainer.train_async(
                this.model, 
                this.dataset, 
                (progress) => this.onTrainingProgress(progress)
            );

            if (this.isTraining) {
                this.updateTrainingStatus('stopped', 'Training completed');
                this.log('Training completed successfully!', 'success');
                this.plotDecisionBoundary();
            }

        } catch (error) {
            this.updateTrainingStatus('error', 'Training failed');
            this.log(`Training failed: ${error}`, 'error');
        } finally {
            this.isTraining = false;
            if (this.elapsedTimer) {
                clearInterval(this.elapsedTimer);
                this.elapsedTimer = null;
            }
            document.getElementById('start-training').disabled = false;
            document.getElementById('stop-training').disabled = true;
        }
    }

    stopTraining() {
        if (this.isTraining) {
            this.isTraining = false;
            if (this.elapsedTimer) {
                clearInterval(this.elapsedTimer);
                this.elapsedTimer = null;
            }
            this.updateTrainingStatus('stopped', 'Training stopped by user');
            this.log('Training stopped by user', 'warning');
        }
    }

    onTrainingProgress(progress) {
        if (!this.isTraining) return;

        // Simulate validation metrics for demo purposes
        const valLoss = progress.loss * (1.0 + Math.random() * 0.2);
        const valAccuracy = progress.accuracy * (0.95 + Math.random() * 0.1);

        // Update trend indicators
        this.updateTrendIndicator('loss-trend', progress.loss, this.lastMetrics.loss, false);
        this.updateTrendIndicator('accuracy-trend', progress.accuracy, this.lastMetrics.accuracy, true);
        this.updateTrendIndicator('val-loss-trend', valLoss, this.lastMetrics.valLoss, false);
        this.updateTrendIndicator('val-accuracy-trend', valAccuracy, this.lastMetrics.valAccuracy, true);
        
        // Update best accuracy tracking
        if (valAccuracy > this.bestAccuracy) {
            this.bestAccuracy = valAccuracy;
            this.bestEpoch = progress.epoch;
            document.getElementById('best-accuracy').textContent = (this.bestAccuracy * 100).toFixed(1) + '%';
            document.getElementById('best-epoch').textContent = this.bestEpoch;
        }

        // Update UI metrics
        document.getElementById('current-epoch').textContent = progress.epoch;
        document.getElementById('current-loss').textContent = progress.loss.toFixed(4);
        document.getElementById('current-accuracy').textContent = (progress.accuracy * 100).toFixed(1) + '%';
        document.getElementById('current-val-loss').textContent = valLoss.toFixed(4);
        document.getElementById('current-val-accuracy').textContent = (valAccuracy * 100).toFixed(1) + '%';
        
        // Update progress bar
        const totalEpochs = parseInt(document.getElementById('epochs').value);
        const progressPercent = (progress.epoch / totalEpochs) * 100;
        document.getElementById('progress-fill').style.width = progressPercent + '%';
        document.getElementById('progress-text').textContent = progressPercent.toFixed(1) + '%';
        
        // Calculate and update ETA
        const elapsed = (Date.now() - this.startTime) / 1000;
        const epochsRemaining = totalEpochs - progress.epoch;
        const avgTimePerEpoch = elapsed / progress.epoch;
        const eta = epochsRemaining * avgTimePerEpoch;
        document.getElementById('eta-time').textContent = eta > 0 ? this.formatTime(eta) : '--:--';

        // Store current metrics for next comparison
        this.lastMetrics = {
            loss: progress.loss,
            accuracy: progress.accuracy,
            valLoss: valLoss,
            valAccuracy: valAccuracy
        };

        // Store history
        this.trainingHistory.push({
            epoch: progress.epoch,
            loss: progress.loss,
            accuracy: progress.accuracy
        });
        
        this.validationHistory.push({
            epoch: progress.epoch,
            loss: valLoss,
            accuracy: valAccuracy
        });

        // Update Chart.js charts
        this.updateCharts();

        this.log(`Epoch ${progress.epoch}: Loss=${progress.loss.toFixed(4)}, Acc=${(progress.accuracy * 100).toFixed(1)}%, Val_Loss=${valLoss.toFixed(4)}, Val_Acc=${(valAccuracy * 100).toFixed(1)}%`, 'info');
    }

    updateCharts() {
        const latestTraining = this.trainingHistory[this.trainingHistory.length - 1];
        const latestValidation = this.validationHistory[this.validationHistory.length - 1];
        
        if (!latestTraining || !latestValidation) return;

        // Update loss chart
        this.charts.loss.data.labels.push(latestTraining.epoch);
        this.charts.loss.data.datasets[0].data.push(latestTraining.loss);
        this.charts.loss.data.datasets[1].data.push(latestValidation.loss);
        
        // Update accuracy chart
        this.charts.accuracy.data.labels.push(latestTraining.epoch);
        this.charts.accuracy.data.datasets[0].data.push(latestTraining.accuracy);
        this.charts.accuracy.data.datasets[1].data.push(latestValidation.accuracy);
        
        // Keep only the last 100 points for performance
        const maxPoints = 100;
        if (this.charts.loss.data.labels.length > maxPoints) {
            this.charts.loss.data.labels.shift();
            this.charts.loss.data.datasets[0].data.shift();
            this.charts.loss.data.datasets[1].data.shift();
            
            this.charts.accuracy.data.labels.shift();
            this.charts.accuracy.data.datasets[0].data.shift();
            this.charts.accuracy.data.datasets[1].data.shift();
        }
        
        // Update charts
        this.charts.loss.update('none'); // 'none' for better performance during training
        this.charts.accuracy.update('none');
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
                    const input = new JsTensor(new Float64Array([x, y]), 1, 2);
                    
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
            
            const input = new JsTensor(new Float64Array([x, y]), 1, 2);
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
