FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Python dependencies. opencv-python-headless wheels are self-contained, so no
# apt packages (libgl1, libsm6, …) are needed — fewer packages, less to patch.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Application code — copied explicitly so nothing else in the checkout
# (.env, .git, notes) can end up in an image layer.
COPY main.py ./
COPY src ./src
COPY static ./static
COPY templates ./templates

# Run unprivileged: a bug in an image decoder should not hand out root in the container.
RUN useradd --system --uid 10001 --no-create-home --shell /usr/sbin/nologin app
USER 10001

EXPOSE 8000

# Single worker on purpose: requests are processed one at a time, which bounds peak memory.
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000", "--proxy-headers", "--no-server-header"]
