ARG AUTHORITY_PLATFORM=linux/amd64
FROM --platform=${AUTHORITY_PLATFORM} python:3.12.10-slim-bookworm@sha256:fd95fa221297a88e1cf49c55ec1828edd7c5a428187e67b5d1805692d11588db

ENV PYTHONHASHSEED=0 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    MUJOCO_GL=egl \
    JAX_ENABLE_X64=True

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        libegl1 \
        libgl1 \
        libglib2.0-0 \
        libosmesa6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /opt/supra
COPY requirements-faithful-lock.txt ./requirements-faithful-lock.txt
RUN python -m pip install --no-cache-dir --disable-pip-version-check --require-hashes \
        -r requirements-faithful-lock.txt

COPY supra ./supra
COPY tools/validate_faithful_runtime.py ./tools/validate_faithful_runtime.py

RUN python tools/validate_faithful_runtime.py --write-identity /opt/supra/runtime-identity.json

ENTRYPOINT ["python", "tools/validate_faithful_runtime.py"]
