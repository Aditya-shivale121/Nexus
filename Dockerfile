FROM node:24-alpine AS frontend-build
WORKDIR /app/frontend
COPY frontend/package*.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM eclipse-temurin:17-jdk-alpine AS java-build
WORKDIR /app
COPY backend/src ./backend/src
RUN javac -d backend/out backend/src/main/java/com/yourownai/App.java

FROM eclipse-temurin:17-jre-alpine
WORKDIR /app
COPY --from=frontend-build /app/frontend/dist ./frontend/dist
COPY --from=java-build /app/backend/out ./backend/out
ENV PORT=8080
EXPOSE 8080
CMD ["sh", "-c", "java -cp backend/out com.yourownai.App ${PORT:-8080}"]
