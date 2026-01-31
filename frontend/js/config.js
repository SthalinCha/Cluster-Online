// Configuración global de la aplicación
const CONFIG = {
    // Detectar automáticamente si estamos en local o producción
    API_BASE: window.location.hostname === 'localhost' || window.location.hostname === '127.0.0.1' 
        ? 'http://localhost:8001'  // Local
        : 'https://cluster-online.onrender.com',  // Producción
    RETRY_ATTEMPTS: 3,
    RETRY_DELAY: 1000,
    DEFAULT_PARAMS: {
        k: 3,
        m: 50,
        threshold: 0.75
    }
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