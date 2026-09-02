# Music DNA Analyzer — standalone web app image.
#   docker build -t music-dna .
#   docker run -p 8000:8000 -v music-data:/data music-dna
FROM python:3.12-slim-bookworm

# System libraries librosa/soundfile/matplotlib need to DECODE the MP3 previews.
# Without ffmpeg + libsndfile1 every track fails to analyze on slim Linux (the
# classic "works on my Windows box" trap). libgomp1 is OpenMP for numpy/numba.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg libsndfile1 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MPLBACKEND=Agg \
    MPLCONFIGDIR=/tmp/mpl \
    NUMBA_CACHE_DIR=/tmp/numba \
    OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
    NUMEXPR_NUM_THREADS=1 NUMBA_NUM_THREADS=1 \
    MUSIC_HOST=0.0.0.0 \
    MUSIC_DATA_DIR=/data

WORKDIR /app

# Install deps first so the slow pip layer is cached across app-code changes.
COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

# Only the code the server needs (see .dockerignore for what is kept out).
COPY app/ ./app/
COPY static/ ./static/

# Run as a non-root user that owns the data volume + writable cache dirs.
RUN useradd -u 10001 -m appuser \
    && mkdir -p /data /tmp/mpl /tmp/numba \
    && chown -R appuser:appuser /app /data /tmp/mpl /tmp/numba
USER appuser

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=40s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8000/healthz',timeout=4).status==200 else 1)"

CMD ["python", "-m", "app"]
