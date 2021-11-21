FROM python:3.9.7-slim

# Set working directory.
WORKDIR /workspace

RUN apt-get -y update && apt-get -y install git openjdk-11-jdk

# Install Pip, Pipenv and Python dependencies.
RUN pip install --upgrade pip && pip install pipenv
COPY Pipfile Pipfile.lock /workspace/
RUN pipenv install --deploy

# Copy source code.
COPY grimjack/ /workspace/grimjack/

# Define entry point for Docker image.
ENTRYPOINT ["pipenv", "run", "python", "-m", "grimjack"]
CMD ["--help"]
