FROM python:3.12-slim

WORKDIR /app

COPY requirements.txt .
# --trusted-host works around SSL-inspecting corporate proxies that break pip's
# default certificate verification; harmless (and a no-op) on a clean network.
RUN pip install --no-cache-dir \
    --trusted-host pypi.org --trusted-host files.pythonhosted.org --trusted-host pypi.python.org \
    -r requirements.txt

COPY src/ src/
COPY dashboard/ dashboard/
COPY models/ models/
COPY results/ results/

EXPOSE 8000 8501

CMD ["uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "8000"]
