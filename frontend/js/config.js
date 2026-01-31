// Configuración global de la aplicación
const CONFIG = {
    // Detectar automáticamente si estamos en local o producción
    API_BASE: (() => {
        const hostname = window.location.hostname;
        
        // Desarrollo local
        if (hostname === 'localhost' || hostname === '127.0.0.1') {
            return 'http://localhost:8001';
        }
        
        // Producción - siempre HTTPS para Render
        return 'https://cluster-online.onrender.com';
    })(),
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000,
    DEFAULT_PARAMS: {
        k: 3,
        m: 50,
        threshold: 0.75
    },
    // Configuración adicional para producción
    PRODUCTION: window.location.hostname !== 'localhost' && window.location.hostname !== '127.0.0.1'
};

// Estado global de la aplicación
const appState = {
    isConnected: false,
    isLoading: false,
    lastConfig: null
};

// Referencias a elementos del DOM
const elements = {
    connectionStatus: null,
    statusText: null,
    loadingOverlay: null,
    paramsResult: null,
    clusteringResult: null,
    inputs: {}
};