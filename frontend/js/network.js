// Utilidades de red para manejo mejorado de errores

// Verificar si estamos online
function isOnline() {
    return navigator.onLine;
}

// Detectar tipo de error de red
function getNetworkErrorType(error) {
    if (!isOnline()) {
        return {
            type: 'offline',
            message: 'Sin conexión a internet',
            suggestion: 'Verifica tu conexión y vuelve a intentar'
        };
    }
    
    if (error.name === 'TypeError' && error.message.includes('fetch')) {
        return {
            type: 'cors',
            message: 'Error de CORS o servidor no disponible',
            suggestion: 'El servidor podría estar arrancando en Render (puede tomar 1-2 minutos)'
        };
    }
    
    if (error.message.includes('502') || error.message.includes('503')) {
        return {
            type: 'server_error',
            message: 'Servidor temporalmente no disponible',
            suggestion: 'Render está arrancando el servicio, intenta en unos segundos'
        };
    }
    
    if (error.message.includes('404')) {
        return {
            type: 'endpoint_error',
            message: 'Endpoint no encontrado',
            suggestion: 'Verifica que el backend esté actualizado'
        };
    }
    
    return {
        type: 'unknown',
        message: error.message || 'Error desconocido',
        suggestion: 'Intenta recargar la página'
    };
}

// Mostrar error mejorado al usuario
function displayNetworkError(error, containerElement) {
    const errorInfo = getNetworkErrorType(error);
    
    const errorHtml = `
        <div class="alert alert-danger network-error">
            <div class="d-flex align-items-center">
                <i class="fas fa-exclamation-triangle me-2"></i>
                <div>
                    <strong>${errorInfo.type === 'offline' ? '🔴 Sin conexión' : '⚠️ Error de conexión'}</strong>
                </div>
            </div>
            <div class="mt-2">
                <p class="mb-1">${errorInfo.message}</p>
                <small class="text-muted">💡 ${errorInfo.suggestion}</small>
            </div>
            ${CONFIG.PRODUCTION ? `
                <div class="mt-2">
                    <small class="text-muted">
                        🔗 Backend: <code>${CONFIG.API_BASE}</code>
                    </small>
                </div>
            ` : ''}
        </div>
    `;
    
    if (containerElement) {
        containerElement.innerHTML = errorHtml;
    }
    
    return errorInfo;
}

// Función mejorada para manejar intentos de reconexión
async function waitForServerReady(maxWaitTime = 120000) { // 2 minutos max
    const startTime = Date.now();
    const checkInterval = 5000; // Cada 5 segundos
    
    while (Date.now() - startTime < maxWaitTime) {
        try {
            const response = await fetch(`${CONFIG.API_BASE}/health`, {
                method: 'GET',
                headers: { 'Accept': 'application/json' },
                mode: 'cors',
                credentials: 'omit'
            });
            
            if (response.ok) {
                return true;
            }
        } catch (error) {
            console.log(`Servidor no listo, reintentando en ${checkInterval/1000}s...`);
        }
        
        await new Promise(resolve => setTimeout(resolve, checkInterval));
    }
    
    return false;
}

// Detectores de eventos de red
window.addEventListener('online', () => {
    showNotification('Conexión restaurada', 'success');
    checkApiConnection(); // Reintentar conexión automáticamente
});

window.addEventListener('offline', () => {
    showNotification('Sin conexión a internet', 'error');
    updateConnectionStatus('error', 'Sin internet');
});

// Exportar funciones para uso global
window.networkUtils = {
    isOnline,
    getNetworkErrorType,
    displayNetworkError,
    waitForServerReady
};