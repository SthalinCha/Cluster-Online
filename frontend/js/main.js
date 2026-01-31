// Funciones principales de la aplicación

// Inicialización de la aplicación
document.addEventListener('DOMContentLoaded', function() {
    initializeElements();
    initializeEventListeners();
    initializeNotificationStyles();
    checkApiConnection();
    
    // Inicializar capacidades personalizadas
    generateCapacityInputs();
    
    // Aplicar animación de entrada con delay
    document.querySelectorAll('.card').forEach((card, index) => {
        card.style.animationDelay = `${index * 0.2}s`;
    });
});

// Función para mostrar/ocultar opciones de capacidades personalizadas
function toggleCapacityOptions() {
    const flexibleSelect = document.getElementById('flexible-select');
    const customSection = document.getElementById('custom-capacities-section');
    
    if (flexibleSelect.value === 'custom') {
        customSection.style.display = 'block';
        customSection.classList.add('show');
        generateCapacityInputs();
    } else {
        customSection.classList.remove('show');
        setTimeout(() => {
            customSection.style.display = 'none';
        }, 400);
    }
}

// Función para generar inputs de capacidad dinámicamente
function generateCapacityInputs() {
    const numClusters = parseInt(document.getElementById('num-clusters-input').value) || 3;
    const capacityInputsContainer = document.getElementById('capacity-inputs');
    
    if (!capacityInputsContainer) return;
    
    capacityInputsContainer.innerHTML = '';
    
    for (let i = 0; i < numClusters; i++) {
        const inputGroup = document.createElement('div');
        inputGroup.className = 'capacity-input-group';
        
        inputGroup.innerHTML = `
            <label for="capacity-${i}">
                <i class="fas fa-circle" style="color: ${getClusterColor(i)}"></i>
                Cluster ${i + 1}
            </label>
            <input 
                type="number" 
                id="capacity-${i}" 
                value="${getDefaultCapacity(i, numClusters)}" 
                min="1" 
                max="200" 
                class="form-control"
                oninput="updateTotalCapacity()"
            >
        `;
        
        capacityInputsContainer.appendChild(inputGroup);
    }
    
    updateTotalCapacity();
}

// Función para obtener color del cluster
function getClusterColor(index) {
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16'];
    return colors[index % colors.length];
}

// Función para obtener capacidad por defecto
function getDefaultCapacity(index, totalClusters) {
    // Ejemplo: para 3 clusters -> 40, 30, 30
    if (totalClusters === 3) {
        return [40, 30, 30][index] || 30;
    }
    // Para otros casos, distribución más equitativa
    const baseCapacity = 30;
    return index === 0 ? baseCapacity + 10 : baseCapacity;
}

// Función para actualizar total de capacidad
function updateTotalCapacity() {
    const totalElement = document.getElementById('total-capacity');
    if (!totalElement) return;
    
    const capacityInputs = document.querySelectorAll('[id^="capacity-"]');
    let total = 0;
    
    capacityInputs.forEach(input => {
        const value = parseInt(input.value);
        // Si el valor no es válido, usar 30 como valor por defecto para el cálculo
        const validValue = isNaN(value) || value <= 0 ? 30 : value;
        total += validValue;
    });
    
    totalElement.textContent = total;
}

// Función para obtener capacidades personalizadas
function getCustomCapacities() {
    const capacityInputs = document.querySelectorAll('[id^="capacity-"]');
    const capacities = [];
    
    capacityInputs.forEach(input => {
        const value = parseInt(input.value);
        // Si el valor no es válido, usar 30 como valor por defecto
        capacities.push(isNaN(value) || value <= 0 ? 30 : value);
    });
    
    return capacities;
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
    const flexibleValue = elements.inputs.flexible.value;
    
    // Preparar request según el modo seleccionado
    let requestBody = {
        model_id: modelId,
        dataset_num: datasetNum
    };
    
    if (flexibleValue === 'custom') {
        // Modo capacidades personalizadas
        const customCapacities = getCustomCapacities();
        
        // Validación adicional por si acaso
        if (customCapacities.length === 0) {
            displayResult(elements.clusteringResult, { 
                error: 'No se encontraron configuraciones de capacidad. Asegúrate de seleccionar el número de clusters primero.' 
            }, 'error');
            return;
        }
        
        console.log('Capacidades personalizadas:', customCapacities);
        
        requestBody.use_custom_capacities = true;
        requestBody.cluster_capacities = customCapacities;
        requestBody.use_flexible = false;  // No aplica en modo custom
        
    } else {
        // Modo normal (flexible o restrictivo)
        requestBody.use_custom_capacities = false;
        requestBody.use_flexible = flexibleValue === 'true';
    }
    
    toggleLoading(true, 'Ejecutando clustering...');
    
    try {
        const startTime = Date.now();
        
        const response = await fetchWithRetry('/cluster', {
            method: 'POST',
            body: JSON.stringify(requestBody)
        });
        
        const result = await response.json();
        const duration = ((Date.now() - startTime) / 1000).toFixed(2);
        
        // Agregar información de tiempo
        result.execution_time_seconds = duration;
        
        let modeDescription;
        if (flexibleValue === 'custom') {
            modeDescription = `Capacidades Personalizadas: ${getCustomCapacities().join(', ')}`;
        } else {
            modeDescription = flexibleValue === 'true' ? 'Flexible' : 'Restrictivo';
        }
        
        displayResult(elements.clusteringResult, {
            message: `Clustering ejecutado correctamente en ${duration}s`,
            parameters: {
                model: modelId,
                dataset: datasetNum,
                mode: modeDescription,
                custom_capacities: flexibleValue === 'custom' ? getCustomCapacities() : null
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
                mode: flexibleValue === 'custom' ? 'Capacidades Personalizadas' : 
                      flexibleValue === 'true' ? 'Flexible' : 'Restrictivo'
            }
        }, 'error');
        
        showNotification('Error en clustering', 'error');
    } finally {
        toggleLoading(false);
    }
}

// Exponer funciones globalmente para usar en HTML
window.updateParams = updateParams;
window.resetParams = resetParams;
window.executeClustering = executeClustering;
window.toggleCapacityOptions = toggleCapacityOptions;
window.generateCapacityInputs = generateCapacityInputs;
window.updateTotalCapacity = updateTotalCapacity;