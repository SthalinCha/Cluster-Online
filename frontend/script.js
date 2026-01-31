// Configuración global
const CONFIG = {
    API_BASE: 'http://localhost:8001',
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000,
    DEFAULT_PARAMS: {
        k: 3,
        m: 50,
        threshold: 0.75
    }
};

// Estado de la aplicación
const appState = {
    isConnected: false,
    isLoading: false,
    lastConfig: null
};

// Elementos del DOM
const elements = {
    connectionStatus: null,
    statusText: null,
    loadingOverlay: null,
    paramsResult: null,
    clusteringResult: null,
    inputs: {}
};

// Inicialización de la aplicación
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    initializeEventListeners();
    checkApiConnection();
    
    // Aplicar animación de entrada con delay
    document.querySelectorAll('.card').forEach((card, index) => {
        card.style.animationDelay = `${index * 0.2}s`;
    });
});

// Inicializar referencias a elementos del DOM
function initializeElements() {
    elements.connectionStatus = document.getElementById('connection-status');
    elements.statusText = elements.connectionStatus?.querySelector('.status-text');
    elements.loadingOverlay = document.getElementById('loading-overlay');
    elements.paramsResult = document.getElementById('params-result');
    elements.clusteringResult = document.getElementById('clustering-result');
    
    // Referencias a inputs
    elements.inputs = {
        k: document.getElementById('k-input'),
        m: document.getElementById('m-input'),
        threshold: document.getElementById('threshold-input'),
        model: document.getElementById('model-select'),
        dataset: document.getElementById('dataset-select'),
        flexible: document.getElementById('flexible-select')
    };
}

// Inicializar event listeners
function initializeEventListeners() {
    // Validación en tiempo real de inputs
    if (elements.inputs.k) {
        elements.inputs.k.addEventListener('input', validateNumberInput);
        elements.inputs.k.addEventListener('blur', () => validateRange(elements.inputs.k, 2, 10));
    }
    
    if (elements.inputs.m) {
        elements.inputs.m.addEventListener('input', validateNumberInput);
        elements.inputs.m.addEventListener('blur', () => validateRange(elements.inputs.m, 10, 200));
    }
    
    if (elements.inputs.threshold) {
        elements.inputs.threshold.addEventListener('input', validateNumberInput);
        elements.inputs.threshold.addEventListener('blur', () => validateRange(elements.inputs.threshold, 0.1, 0.99));
    }
    
    // Shortcuts de teclado
    document.addEventListener('keydown', handleKeyboardShortcuts);
    
    // Prevenir envío accidental de formularios
    document.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.target.tagName === 'INPUT') {
            e.preventDefault();
        }
    });
}

// Validar inputs numéricos
function validateNumberInput(event) {
    const input = event.target;
    const value = parseFloat(input.value);
    
    if (isNaN(value) && input.value !== '') {
        input.classList.add('is-invalid');
        showTooltip(input, 'Ingresa un número válido');
    } else {
        input.classList.remove('is-invalid');
        hideTooltip(input);
    }
}

// Validar rangos
function validateRange(input, min, max) {
    const value = parseFloat(input.value);
    
    if (value < min || value > max) {
        input.classList.add('is-invalid');
        showTooltip(input, `Valor debe estar entre ${min} y ${max}`);
        return false;
    } else {
        input.classList.remove('is-invalid');
        hideTooltip(input);
        return true;
    }
}

// Mostrar tooltip de error
function showTooltip(element, message) {
    let tooltip = element.parentNode.querySelector('.tooltip-error');
    
    if (!tooltip) {
        tooltip = document.createElement('div');
        tooltip.className = 'tooltip-error';
        tooltip.style.cssText = `
            position: absolute;
            background: var(--danger);
            color: white;
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 12px;
            z-index: 1000;
            margin-top: 4px;
            animation: slideInUp 0.2s ease;
        `;
        element.parentNode.style.position = 'relative';
        element.parentNode.appendChild(tooltip);
    }
    
    tooltip.textContent = message;
}

// Ocultar tooltip de error
function hideTooltip(element) {
    const tooltip = element.parentNode.querySelector('.tooltip-error');
    if (tooltip) {
        tooltip.remove();
    }
}

// Manejar shortcuts de teclado
function handleKeyboardShortcuts(event) {
    if (event.ctrlKey || event.metaKey) {
        switch (event.key) {
            case 'u':
                event.preventDefault();
                updateParams();
                break;
            case 'e':
                event.preventDefault();
                if (!appState.isLoading) {
                    executeClustering();
                }
                break;
            case 'r':
                event.preventDefault();
                resetParams();
                break;
        }
    }
}

// Verificar conexión con la API
async function checkApiConnection() {
    try {
        updateConnectionStatus('connecting', 'Conectando...');
        
        const response = await fetchWithRetry('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            appState.isConnected = true;
            updateConnectionStatus('connected', 'Conectado');
            
            // Cargar configuración inicial
            await loadCurrentConfig();
        } else {
            throw new Error('API no saludable');
        }
    } catch (error) {
        appState.isConnected = false;
        updateConnectionStatus('error', 'Error de conexión');
        console.error('Error de conexión con API:', error);
        
        // Reintentar conexión después de 5 segundos
        setTimeout(checkApiConnection, 5000);
    }
}

// Actualizar estado de conexión
function updateConnectionStatus(status, text) {
    if (!elements.connectionStatus || !elements.statusText) return;
    
    elements.connectionStatus.className = `status ${status}`;
    elements.statusText.textContent = text;
}

// Cargar configuración actual
async function loadCurrentConfig() {
    try {
        const response = await fetchWithRetry('/config');
        const config = await response.json();
        
        appState.lastConfig = config;
        
        // Actualizar inputs con la configuración actual
        if (elements.inputs.k) elements.inputs.k.value = config.k || CONFIG.DEFAULT_PARAMS.k;
        if (elements.inputs.m) elements.inputs.m.value = config.m || CONFIG.DEFAULT_PARAMS.m;
        if (elements.inputs.threshold) elements.inputs.threshold.value = config.cluster_similarity_threshold || CONFIG.DEFAULT_PARAMS.threshold;
        
    } catch (error) {
        console.warn('No se pudo cargar la configuración actual:', error);
    }
}

// Función para hacer fetch con reintentos
async function fetchWithRetry(endpoint, options = {}, attempts = CONFIG.RETRY_ATTEMPTS) {
    for (let i = 0; i < attempts; i++) {
        try {
            const response = await fetch(`${CONFIG.API_BASE}${endpoint}`, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    ...options.headers
                }
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return response;
        } catch (error) {
            console.warn(`Intento ${i + 1} fallido:`, error.message);
            
            if (i === attempts - 1) {
                throw error;
            }
            
            await new Promise(resolve => setTimeout(resolve, CONFIG.RETRY_DELAY));
        }
    }
}

// Mostrar/ocultar overlay de carga
function toggleLoading(show, message = 'Procesando...') {
    if (!elements.loadingOverlay) return;
    
    appState.isLoading = show;
    
    if (show) {
        elements.loadingOverlay.querySelector('p').textContent = message;
        elements.loadingOverlay.classList.add('active');
    } else {
        elements.loadingOverlay.classList.remove('active');
    }
}

// Mostrar resultado en contenedor
function displayResult(container, data, type = 'info') {
    if (!container) return;
    
    const resultClass = `result-${type}`;
    const iconClass = type === 'success' ? 'fa-check-circle' : 
                     type === 'error' ? 'fa-exclamation-circle' : 
                     'fa-info-circle';
    
    let content = '';
    
    if (type === 'success' && data.config && data.config.updated_parameters) {
        // Para resultados de actualización de parámetros
        content = createParamsResultHTML(data);
    } else if (type === 'success' && data.results && data.results.clustering_results) {
        // Para resultados de clustering
        content = createClusteringResultHTML(data);
    } else {
        // Para otros tipos de resultados
        content = `<pre>${JSON.stringify(data, null, 2)}</pre>`;
    }
    
    container.innerHTML = `
        <div class="${resultClass}">
            <div style="display: flex; align-items: center; gap: 8px; margin-bottom: 16px;">
                <i class="fas ${iconClass}"></i>
                <strong>${type === 'success' ? 'Éxito' : type === 'error' ? 'Error' : 'Información'}</strong>
            </div>
            ${content}
        </div>
    `;
    
    // Scroll suave al resultado
    container.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
}

// Crear HTML para resultados de parámetros
function createParamsResultHTML(data) {
    const params = data.config.updated_parameters;
    return `
        <div class="result-card">
            <h4><i class="fas fa-cogs"></i> Parámetros Actualizados</h4>
            <div class="params-grid">
                <div class="param-item">
                    <span class="param-label">Clusters (k):</span>
                    <span class="param-value">${params.k}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Capacidad (m):</span>
                    <span class="param-value">${params.m}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Umbral similitud:</span>
                    <span class="param-value">${params.cluster_similarity_threshold}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Sub-umbral:</span>
                    <span class="param-value">${params.subcluster_similarity_threshold}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Máximo pares:</span>
                    <span class="param-value">${params.pair_similarity_maximum}</span>
                </div>
                <div class="param-item">
                    <span class="param-label">Normalización:</span>
                    <span class="param-value ${params.normalize_data ? 'enabled' : 'disabled'}">
                        ${params.normalize_data ? 'Activada' : 'Desactivada'}
                    </span>
                </div>
                <div class="param-item">
                    <span class="param-label">PCA:</span>
                    <span class="param-value ${params.apply_pca ? 'enabled' : 'disabled'}">
                        ${params.apply_pca ? `Activado (${params.pca_components} comp.)` : 'Desactivado'}
                    </span>
                </div>
            </div>
        </div>
    `;
}

// Crear HTML para resultados de clustering
function createClusteringResultHTML(data) {
    const results = data.results;
    const clustering = results.clustering_results;
    const metrics = results.metrics;
    
    return `
        <div class="clustering-results">
            <div class="result-header">
                <h4><i class="fas fa-project-diagram"></i> Resultado del Clustering</h4>
                <span class="execution-time">${data.results.execution_time_seconds}s</span>
            </div>
            
            <div class="result-grid">
                <!-- Información del modelo -->
                <div class="info-card">
                    <h5><i class="fas fa-brain"></i> Modelo</h5>
                    <div class="info-content">
                        <p><strong>${results.model.name}</strong></p>
                        <p class="description">${results.model.description}</p>
                        <p class="algorithm">${results.algorithm}</p>
                    </div>
                </div>
                
                <!-- Información de datos -->
                <div class="info-card">
                    <h5><i class="fas fa-database"></i> Datos</h5>
                    <div class="info-content">
                        <p>Dataset: <strong>${results.dataset}</strong></p>
                        <p>Forma original: <strong>${results.data_info.original_shape[0]} × ${results.data_info.original_shape[1]}</strong></p>
                        <p>Procesados: <strong>${results.data_info.processed_shape[0]} × ${results.data_info.processed_shape[1]}</strong></p>
                        <p>Clases reales: <strong>${results.data_info.true_classes}</strong></p>
                    </div>
                </div>
                
                <!-- Resultados del clustering -->
                <div class="info-card">
                    <h5><i class="fas fa-layer-group"></i> Clusters Formados</h5>
                    <div class="clusters-info">
                        <div class="cluster-summary">
                            <div class="cluster-count">
                                <span class="number">${clustering.n_clusters_formed}</span>
                                <span class="label">Clusters</span>
                            </div>
                            <div class="sample-count">
                                <span class="number">${clustering.total_samples}</span>
                                <span class="label">Muestras</span>
                            </div>
                        </div>
                        <div class="cluster-sizes">
                            ${clustering.cluster_sizes.map((size, i) => `
                                <div class="cluster-bar">
                                    <span class="cluster-label">Cluster ${i}</span>
                                    <div class="bar-container">
                                        <div class="bar" style="width: ${(size / Math.max(...clustering.cluster_sizes)) * 100}%"></div>
                                        <span class="size">${size}</span>
                                    </div>
                                </div>
                            `).join('')}
                        </div>
                    </div>
                </div>
                
                <!-- Métricas -->
                <div class="info-card metrics-card">
                    <h5><i class="fas fa-chart-line"></i> Métricas de Evaluación</h5>
                    <div class="metrics-grid">
                        <div class="metric-group">
                            <h6>Externas</h6>
                            <div class="metric-item">
                                <span class="metric-name">ARI</span>
                                <span class="metric-value">${metrics.external.ARI.toFixed(4)}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-name">NMI</span>
                                <span class="metric-value">${metrics.external.NMI.toFixed(4)}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-name">AMI</span>
                                <span class="metric-value">${metrics.external.AMI.toFixed(4)}</span>
                            </div>
                        </div>
                        
                        <div class="metric-group">
                            <h6>Internas</h6>
                            <div class="metric-item">
                                <span class="metric-name">Silhouette</span>
                                <span class="metric-value">${metrics.internal.Silhouette.toFixed(4)}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-name">Davies-Bouldin</span>
                                <span class="metric-value">${metrics.internal.Davies_Bouldin.toFixed(4)}</span>
                            </div>
                            <div class="metric-item">
                                <span class="metric-name">Calinski-Harabasz</span>
                                <span class="metric-value">${metrics.internal.Calinski_Harabasz.toFixed(2)}</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>
    `;
}

// Validar formulario completo
function validateForm() {
    const validations = [
        validateRange(elements.inputs.k, 2, 10),
        validateRange(elements.inputs.m, 10, 200),
        validateRange(elements.inputs.threshold, 0.1, 0.99)
    ];
    
    return validations.every(valid => valid);
}

// Actualizar parámetros
async function updateParams() {
    if (!appState.isConnected) {
        displayResult(elements.paramsResult, { error: 'Sin conexión a la API' }, 'error');
        return;
    }
    
    if (!validateForm()) {
        displayResult(elements.paramsResult, { error: 'Valores inválidos en el formulario' }, 'error');
        return;
    }
    
    const k = parseInt(elements.inputs.k.value);
    const m = parseInt(elements.inputs.m.value);
    const threshold = parseFloat(elements.inputs.threshold.value);
    
    toggleLoading(true, 'Actualizando parámetros...');
    
    try {
        const response = await fetchWithRetry('/config', {
            method: 'PUT',
            body: JSON.stringify({
                k: k,
                m: m,
                cluster_similarity_threshold: threshold
            })
        });
        
        const result = await response.json();
        appState.lastConfig = result;
        
        displayResult(elements.paramsResult, {
            message: 'Parámetros actualizados correctamente',
            config: result
        }, 'success');
        
        // Notificación visual
        showNotification('Parámetros actualizados', 'success');
        
    } catch (error) {
        console.error('Error actualizando parámetros:', error);
        displayResult(elements.paramsResult, {
            error: error.message,
            details: 'No se pudieron actualizar los parámetros'
        }, 'error');
        
        showNotification('Error al actualizar parámetros', 'error');
    } finally {
        toggleLoading(false);
    }
}

// Restablecer parámetros
function resetParams() {
    elements.inputs.k.value = CONFIG.DEFAULT_PARAMS.k;
    elements.inputs.m.value = CONFIG.DEFAULT_PARAMS.m;
    elements.inputs.threshold.value = CONFIG.DEFAULT_PARAMS.threshold;
    
    // Limpiar validaciones
    Object.values(elements.inputs).forEach(input => {
        if (input) {
            input.classList.remove('is-invalid');
            hideTooltip(input);
        }
    });
    
    // Limpiar resultado
    if (elements.paramsResult) {
        elements.paramsResult.innerHTML = '';
    }
    
    showNotification('Parámetros restablecidos', 'info');
}

// Ejecutar clustering
async function executeClustering() {
    if (!appState.isConnected) {
        displayResult(elements.clusteringResult, { error: 'Sin conexión a la API' }, 'error');
        return;
    }
    
    if (appState.isLoading) {
        showNotification('Ya hay una operación en progreso', 'warning');
        return;
    }
    
    const modelId = elements.inputs.model.value;
    const datasetNum = parseInt(elements.inputs.dataset.value);
    const useFlexible = elements.inputs.flexible.value === 'true';
    
    toggleLoading(true, 'Ejecutando clustering...');
    
    try {
        const startTime = Date.now();
        
        const response = await fetchWithRetry('/cluster', {
            method: 'POST',
            body: JSON.stringify({
                model_id: modelId,
                dataset_num: datasetNum,
                use_flexible: useFlexible
            })
        });
        
        const result = await response.json();
        const duration = ((Date.now() - startTime) / 1000).toFixed(2);
        
        // Agregar información de tiempo
        result.execution_time_seconds = duration;
        
        displayResult(elements.clusteringResult, {
            message: `Clustering ejecutado correctamente en ${duration}s`,
            parameters: {
                model: modelId,
                dataset: datasetNum,
                flexible: useFlexible
            },
            results: result
        }, 'success');
        
        showNotification(`Clustering completado en ${duration}s`, 'success');
        
    } catch (error) {
        console.error('Error ejecutando clustering:', error);
        displayResult(elements.clusteringResult, {
            error: error.message,
            details: 'No se pudo ejecutar el clustering',
            parameters: {
                model: modelId,
                dataset: datasetNum,
                flexible: useFlexible
            }
        }, 'error');
        
        showNotification('Error en clustering', 'error');
    } finally {
        toggleLoading(false);
    }
}

// Sistema de notificaciones
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        background: ${type === 'success' ? 'var(--success)' : 
                    type === 'error' ? 'var(--danger)' : 
                    type === 'warning' ? 'var(--warning)' : 'var(--info)'};
        color: white;
        padding: 12px 20px;
        border-radius: var(--border-radius);
        box-shadow: var(--shadow-lg);
        z-index: 10000;
        animation: slideInRight 0.3s ease;
        max-width: 300px;
        word-wrap: break-word;
    `;
    
    const icon = type === 'success' ? 'fa-check' : 
                type === 'error' ? 'fa-times' : 
                type === 'warning' ? 'fa-exclamation-triangle' : 'fa-info';
    
    notification.innerHTML = `
        <div style="display: flex; align-items: center; gap: 8px;">
            <i class="fas ${icon}"></i>
            <span>${message}</span>
        </div>
    `;
    
    document.body.appendChild(notification);
    
    // Auto-remover notificación
    setTimeout(() => {
        notification.style.animation = 'slideOutRight 0.3s ease';
        setTimeout(() => {
            if (notification.parentNode) {
                notification.parentNode.removeChild(notification);
            }
        }, 300);
    }, duration);
}
const notificationStyles = document.createElement('style');
notificationStyles.textContent = `
    @keyframes slideInRight {
        from {
            transform: translateX(100%);
            opacity: 0;
        }
        to {
            transform: translateX(0);
            opacity: 1;
        }
    }
    
    @keyframes slideOutRight {
        from {
            transform: translateX(0);
            opacity: 1;
        }
        to {
            transform: translateX(100%);
            opacity: 0;
        }
    }
    
    .form-control.is-invalid {
        border-color: var(--danger);
        box-shadow: 0 0 0 3px rgba(239, 68, 68, 0.1);
    }
`;
document.head.appendChild(notificationStyles);

// Exponer funciones globalmente para usar en HTML
window.updateParams = updateParams;
window.resetParams = resetParams;
window.executeClustering = executeClustering;