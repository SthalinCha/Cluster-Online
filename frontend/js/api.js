// Funciones para comunicación con la API

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