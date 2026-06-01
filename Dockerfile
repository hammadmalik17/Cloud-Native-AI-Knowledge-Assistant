# Use slim Python image — smaller attack surface, faster pulls
FROM python:3.11-slim

# Set working directory inside the container
WORKDIR /app

# Install dependencies first — Docker layer caching means this layer
# only rebuilds when requirements.txt changes, not on every code change
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY app/ ./app/

# Create uploads directory inside container
RUN mkdir -p uploads

# Don't run as root — security best practice
RUN adduser --disabled-password --gecos "" appuser && chown -R appuser /app
USER appuser

# Expose the port uvicorn will listen on
EXPOSE 8000

# Start FastAPI with uvicorn
# --host 0.0.0.0 makes it reachable from outside the container
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]