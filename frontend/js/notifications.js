// Sistema de notificaciones

// Mostrar notificación
function showNotification(message, type = 'info', duration = 3000) {
    const notification = document.createElement('div');
    notification.className = `notification notification-${type}`;
    
    // Determinar color de fondo
    let backgroundColor;
    if (type === 'success') backgroundColor = '#10b981';
    else if (type === 'error') backgroundColor = '#ef4444';
    else if (type === 'warning') backgroundColor = '#f59e0b';
    else backgroundColor = '#06b6d4';
    
    notification.style.cssText = [
        'position: fixed',
        'top: 20px',
        'right: 20px',
        `background: ${backgroundColor}`,
        'color: white',
        'padding: 12px 20px',
        'border-radius: 0.375rem',
        'box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.1), 0 4px 6px -2px rgba(0, 0, 0, 0.05)',
        'z-index: 10000',
        'animation: slideInRight 0.3s ease',
        'max-width: 300px',
        'word-wrap: break-word'
    ].join('; ');
    
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

// Agregar estilos de animación para notificaciones
function initializeNotificationStyles() {
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
}