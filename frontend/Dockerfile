FROM nginx:alpine

# Copiar todos los archivos del frontend
COPY index.html /usr/share/nginx/html/index.html
COPY styles.css /usr/share/nginx/html/styles.css
COPY js/ /usr/share/nginx/html/js/

# Exponer puerto 80
EXPOSE 80

# Nginx se ejecuta automáticamente