FROM python:3.9.7-slim

# Set working directory.
WORKDIR /workspace

# Install Pip, Pipenv and Python dependencies.
RUN pip install --upgrade pip && pip install pipenv
COPY Pipfile Pipfile.lock /workspace/
RUN pipenv
RUN pipenv install --deploy

# Copy source code.
COPY grimjack/ /workspace/grimjack/

# Define entry point for Docker image.
ENTRYPOINT ["python", "-m", "grimjack.run"]
CMD ["--help"]