FROM nginx:1.27-alpine
COPY index.html /usr/share/nginx/html/index.html
COPY app.js /usr/share/nginx/html/app.js
COPY styles.css /usr/share/nginx/html/styles.css
RUN printf 'server {\n  listen 8080;\n  location / { root /usr/share/nginx/html; index index.html; }\n}\n' > /etc/nginx/conf.d/default.conf
EXPOSE 8080
CMD ["nginx", "-g", "daemon off;"]
