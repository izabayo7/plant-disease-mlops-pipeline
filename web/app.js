// API Configuration
const API_BASE_URL = 'http://localhost:8000';

// DOM Elements
const fileInput = document.getElementById('file-input');
const uploadArea = document.getElementById('upload-area');
const predictBtn = document.getElementById('predict-btn');
const results = document.getElementById('results');
const resultsContainer = document.getElementById('results-container');
const retrainBtn = document.getElementById('retrain-btn');
const retrainStatus = document.getElementById('retrain-status');

// Metrics Elements
const uptimeEl = document.getElementById('uptime');
const totalRequestsEl = document.getElementById('total-requests');
const avgInferenceEl = document.getElementById('avg-inference');
const cpuUsageEl = document.getElementById('cpu-usage');

// Retraining Elements
const uploadDataForm = document.getElementById('upload-data-form');
const uploadStatus = document.getElementById('upload-status');

// Dashboard Elements
const trainChartCtx = document.getElementById('trainChart').getContext('2d');
const newDataChartCtx = document.getElementById('newDataChart').getContext('2d');
let trainChartInstance = null;
let newDataChartInstance = null;

// State
let selectedFiles = [];

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    setupEventListeners();
    startMonitoring();
});

function setupEventListeners() {
    // File input
    uploadArea.addEventListener('click', () => fileInput.click());
    fileInput.addEventListener('change', handleFileSelect);

    // Drag and drop
    uploadArea.addEventListener('dragover', handleDragOver);
    uploadArea.addEventListener('dragleave', handleDragLeave);
    uploadArea.addEventListener('drop', handleDrop);

    // Predict button
    predictBtn.addEventListener('click', handlePredict);

    // Retrain button
    retrainBtn.addEventListener('click', handleRetrain);

    // Auto-fill label from zip filename
    document.getElementById('retrain-file').addEventListener('change', function (e) {
        const file = e.target.files[0];
        if (file && file.name.endsWith('.zip')) {
            // Extract label from filename
            // Example: "Banana___Cordana.zip" -> "Banana___Cordana"
            let label = file.name.replace('.zip', '');

            // Remove common prefixes if present
            label = label.replace(/^(data\/|images\/|photos\/)/i, '');

            // Auto-populate the label field
            document.getElementById('retrain-label').value = label;
        }
    });

    // Upload Data Form
    uploadDataForm.addEventListener('submit', handleUploadData);

    // Tabs
    document.querySelectorAll('.nav-tab').forEach(tab => {
        tab.addEventListener('click', () => {
            // Remove active class from all tabs and contents
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));

            // Add active class to clicked tab and corresponding content
            tab.classList.add('active');
            document.getElementById(tab.dataset.tab).classList.add('active');

            // Load stats if dashboard is selected
            if (tab.dataset.tab === 'dashboard') {
                loadDashboard();
            }
        });
    });
}

function handleFileSelect(e) {
    const files = Array.from(e.target.files);
    selectedFiles = files;
    updateUploadUI();
}

function handleDragOver(e) {
    e.preventDefault();
    uploadArea.classList.add('drag-over');
}

function handleDragLeave(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');
}

function handleDrop(e) {
    e.preventDefault();
    uploadArea.classList.remove('drag-over');

    const files = Array.from(e.dataTransfer.files);
    selectedFiles = files.filter(file => file.type.startsWith('image/'));
    fileInput.files = createFileList(selectedFiles);
    updateUploadUI();
}

function createFileList(files) {
    const dataTransfer = new DataTransfer();
    files.forEach(file => dataTransfer.items.add(file));
    return dataTransfer.files;
}

function updateUploadUI() {
    if (selectedFiles.length > 0) {
        uploadArea.querySelector('.upload-prompt p').textContent =
            `${selectedFiles.length} file(s) selected`;
        predictBtn.disabled = false;
    } else {
        uploadArea.querySelector('.upload-prompt p').textContent =
            'Click to upload or drag and drop';
        predictBtn.disabled = true;
    }
}

async function handlePredict() {
    if (selectedFiles.length === 0) return;

    predictBtn.disabled = true;
    predictBtn.textContent = 'Analyzing...';
    resultsContainer.innerHTML = '';
    results.classList.remove('hidden');

    try {
        const formData = new FormData();
        selectedFiles.forEach(file => formData.append('files', file));

        const response = await fetch(`${API_BASE_URL}/predict-batch`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        if (data.success) {
            displayResults(data.predictions);
        } else {
            showError('Prediction failed');
        }
    } catch (error) {
        console.error('Error:', error);
        showError('Network error. Please try again.');
    } finally {
        predictBtn.disabled = false;
        predictBtn.textContent = 'Analyze Images';
    }
}

function displayResults(predictions) {
    resultsContainer.innerHTML = '';

    predictions.forEach((pred, index) => {
        const card = document.createElement('div');
        card.className = 'result-card';

        const file = selectedFiles[index];
        const imageUrl = URL.createObjectURL(file);

        card.innerHTML = `
            <img src="${imageUrl}" alt="${pred.disease}">
            <div class="result-info">
                <div class="result-disease">${pred.disease}</div>
                <div class="result-confidence">Confidence: ${(pred.confidence * 100).toFixed(2)}%</div>
                <div class="confidence-bar">
                    <div class="confidence-fill" style="width: ${pred.confidence * 100}%"></div>
                </div>
                <div class="result-time">Inference: ${pred.inference_time_ms}ms</div>
            </div>
        `;

        resultsContainer.appendChild(card);
    });
}

async function startMonitoring() {
    // Update monitoring metrics every 5 seconds
    setInterval(updateMonitoring, 5000);
    updateMonitoring(); // Initial update
}

async function updateMonitoring() {
    try {
        const response = await fetch(`${API_BASE_URL}/monitoring`);
        const data = await response.json();

        uptimeEl.textContent = data.uptime_human;
        totalRequestsEl.textContent = data.total_requests.toLocaleString();
        avgInferenceEl.textContent = `${data.inference_metrics.average_ms.toFixed(1)}ms`;
        cpuUsageEl.textContent = `${data.system.cpu_percent.toFixed(1)}%`;
    } catch (error) {
        console.error('Monitoring error:', error);
    }
}

async function handleRetrain() {
    retrainBtn.disabled = true;
    retrainBtn.textContent = 'Initiating...';

    try {
        const response = await fetch(`${API_BASE_URL}/retrain`, {
            method: 'POST'
        });

        const data = await response.json();

        retrainStatus.classList.remove('hidden');
        retrainStatus.classList.add('success');
        retrainStatus.textContent = data.message;
    } catch (error) {
        retrainStatus.classList.remove('hidden');
        retrainStatus.classList.add('error');
        retrainStatus.textContent = 'Failed to initiate retraining';
    } finally {
        retrainBtn.disabled = false;
        retrainBtn.textContent = 'Start Retraining Job';
    }
}

async function handleUploadData(e) {
    e.preventDefault();
    const label = document.getElementById('retrain-label').value;
    const file = document.getElementById('retrain-file').files[0];

    if (!label || !file) return;

    const formData = new FormData();
    formData.append('label', label);
    formData.append('file', file);

    uploadStatus.classList.remove('hidden');
    uploadStatus.textContent = 'Uploading...';
    uploadStatus.className = 'status-message'; // Reset classes

    try {
        const response = await fetch(`${API_BASE_URL}/upload-data`, {
            method: 'POST',
            body: formData
        });

        const data = await response.json();

        uploadStatus.classList.add('success');
        uploadStatus.textContent = data.message;
        uploadDataForm.reset();
    } catch (error) {
        uploadStatus.classList.add('error');
        uploadStatus.textContent = 'Upload failed';
        console.error(error);
    }
}

async function loadDashboard() {
    try {
        const response = await fetch(`${API_BASE_URL}/stats`);
        const stats = await response.json();

        renderCharts(stats);
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

function renderCharts(stats) {
    // Training Data Chart
    if (trainChartInstance) trainChartInstance.destroy();

    const trainLabels = Object.keys(stats.training_data);
    const trainData = Object.values(stats.training_data);

    trainChartInstance = new Chart(trainChartCtx, {
        type: 'bar',
        data: {
            labels: trainLabels,
            datasets: [{
                label: 'Images per Class',
                data: trainData,
                backgroundColor: '#2ecc71'
            }]
        },
        options: {
            responsive: true,
            plugins: { title: { display: true, text: `Total Training Images: ${stats.total_images}` } }
        }
    });

    // New Data Chart
    if (newDataChartInstance) newDataChartInstance.destroy();

    const newLabels = Object.keys(stats.new_data);
    const newData = Object.values(stats.new_data);

    if (newLabels.length > 0) {
        newDataChartInstance = new Chart(newDataChartCtx, {
            type: 'doughnut',
            data: {
                labels: newLabels,
                datasets: [{
                    data: newData,
                    backgroundColor: ['#3498db', '#e74c3c', '#f1c40f', '#9b59b6']
                }]
            },
            options: { responsive: true }
        });
    }
}

function showError(message) {
    resultsContainer.innerHTML = `
        <div class="status-message error">
            ${message}
        </div>
    `;
}
