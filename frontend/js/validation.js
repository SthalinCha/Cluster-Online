// Funciones para validación de formularios

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

// Validar formulario completo
function validateForm() {
    const validations = [
        validateRange(elements.inputs.k, 2, 10),
        validateRange(elements.inputs.m, 10, 200),
        validateRange(elements.inputs.threshold, 0.1, 0.99)
    ];
    
    return validations.every(valid => valid);
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