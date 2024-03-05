FROM openjdk:11-slim as openjdk-11

FROM python:3.9-slim as python

# Install JDK.
COPY --from=openjdk-11 /usr/local/openjdk-11 /usr/local/openjdk
ENV JAVA_HOME /usr/local/openjdk
RUN update-alternatives --install /usr/bin/java java /usr/local/openjdk/bin/java 1

# Install Git and GCC.
RUN --mount=type=cache,target=/var/cache/apt \
    apt-get update -y && \
    apt-get install -y git build-essential

# Install Pip.
RUN --mount=type=cache,target=/root/.cache/pip \
    ([ -d /venv ] || python3.9 -m venv /venv) && \
    /venv/bin/pip install --upgrade pip

# Set working directory.
WORKDIR /workspace/

# Install Python dependencies.
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True
ADD pyproject.toml pyproject.toml
ARG PSEUDO_VERSION=1
RUN --mount=type=cache,target=/root/.cache/pip \
    SETUPTOOLS_SCM_PRETEND_VERSION=${PSEUDO_VERSION} \
    /venv/bin/pip install -e .
RUN --mount=source=.git,target=.git,type=bind \
    --mount=type=cache,target=/root/.cache/pip \
    /venv/bin/pip install -e .

# Copy source code.
COPY grimjack/ /workspace/grimjack/

# Define entry point for Docker image.
ENTRYPOINT ["/venv/bin/python", "-m", "grimjack"]
CMD ["--help"]
