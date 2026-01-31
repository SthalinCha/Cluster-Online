// Funciones para comunicación con la API

// Función para hacer fetch con reintentos
async function fetchWithRetry(endpoint, options = {}, attempts = CONFIG.RETRY_ATTEMPTS) {
    for (let i = 0; i < attempts; i++) {
        try {
            const response = await fetch(`${CONFIG.API_BASE}${endpoint}`, {
                ...options,
                headers: {
                    'Content-Type': 'application/json',
                    'Accept': 'application/json',
                    ...options.headers
                },
                // Configuración para CORS cuando esté en producción
                mode: 'cors',
                credentials: 'omit'
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}: ${response.statusText}`);
            }
            
            return response;
        } catch (error) {
            console.warn(`Intento ${i + 1} fallido para ${endpoint}:`, error.message);
            
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
        
        // Si estamos en producción y el servidor podría estar cold, esperar
        if (CONFIG.PRODUCTION) {
            showNotification('Iniciando conexión con Render...', 'info');
        }
        
        const response = await fetchWithRetry('/health');
        const data = await response.json();
        
        if (data.status === 'healthy') {
            appState.isConnected = true;
            updateConnectionStatus('connected', 'Conectado');
            
            // Cargar configuración inicial
            await loadCurrentConfig();
            
            if (CONFIG.PRODUCTION) {
                showNotification('✅ Conectado a servidor en la nube', 'success');
            }
        } else {
            throw new Error('API no saludable');
        }
    } catch (error) {
        appState.isConnected = false;
        updateConnectionStatus('error', 'Error de conexión');
        
        // Usar utilidades de red para mostrar error detallado
        if (window.networkUtils) {
            const errorInfo = window.networkUtils.getNetworkErrorType(error);
            
            // Si es error de servidor en producción, intentar esperar
            if (CONFIG.PRODUCTION && (errorInfo.type === 'cors' || errorInfo.type === 'server_error')) {
                showNotification('⏳ Servidor arrancando en Render, esperando...', 'warning');
                
                const serverReady = await window.networkUtils.waitForServerReady(120000);
                if (serverReady) {
                    return checkApiConnection(); // Reintentar
                }
            }
        }
        
        console.error('Error de conexión con API:', error);
        
        // Reintentar conexión después de 10 segundos en producción, 5 en local
        const retryDelay = CONFIG.PRODUCTION ? 10000 : 5000;
        setTimeout(checkApiConnection, retryDelay);
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