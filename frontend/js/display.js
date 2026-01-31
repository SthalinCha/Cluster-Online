// Funciones para mostrar resultados de forma atractiva

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
    const hasCustomCapacities = clustering.custom_capacities || clustering.target_capacities;
    
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
                            ${clustering.cluster_sizes.map((size, i) => {
                                const targetCapacity = hasCustomCapacities ? 
                                    (clustering.target_capacities && clustering.target_capacities[i]) || 
                                    (clustering.custom_capacities && clustering.custom_capacities[i]) : 
                                    null;
                                
                                const utilizationText = targetCapacity ? 
                                    ` / ${targetCapacity} (${((size / targetCapacity) * 100).toFixed(1)}%)` : '';
                                
                                const barWidth = hasCustomCapacities && targetCapacity ? 
                                    (size / targetCapacity) * 100 : 
                                    (size / Math.max(...clustering.cluster_sizes)) * 100;
                                
                                const barColor = hasCustomCapacities && targetCapacity ? 
                                    (size <= targetCapacity ? '#10b981' : '#ef4444') : 
                                    'linear-gradient(135deg, var(--primary), var(--primary-dark))';
                                
                                return `
                                    <div class="cluster-bar">
                                        <span class="cluster-label">Cluster ${i}</span>
                                        <div class="bar-container">
                                            <div class="bar" style="width: ${Math.min(barWidth, 100)}%; background: ${barColor}"></div>
                                            <span class="size">${size}${utilizationText}</span>
                                        </div>
                                    </div>
                                `;
                            }).join('')}
                        </div>
                        
                        ${hasCustomCapacities ? `
                            <div class="capacity-summary">
                                <h6><i class="fas fa-info-circle"></i> Capacidades Configuradas</h6>
                                <div class="capacity-list">
                                    ${(clustering.target_capacities || clustering.custom_capacities).map((cap, i) => `
                                        <span class="capacity-item">
                                            <i class="fas fa-circle" style="color: ${getClusterColorForDisplay(i)}"></i>
                                            Cluster ${i}: ${cap}
                                        </span>
                                    `).join('')}
                                </div>
                            </div>
                        ` : ''}
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

// Función auxiliar para obtener colores de cluster en display
function getClusterColorForDisplay(index) {
    const colors = ['#3b82f6', '#10b981', '#f59e0b', '#ef4444', '#8b5cf6', '#06b6d4', '#f97316', '#84cc16'];
    return colors[index % colors.length];
}