// Funciones para manejo de la interfaz de usuario

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

// Actualizar estado de conexión
function updateConnectionStatus(status, text) {
    if (!elements.connectionStatus || !elements.statusText) return;
    
    elements.connectionStatus.className = `status ${status}`;
    elements.statusText.textContent = text;
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