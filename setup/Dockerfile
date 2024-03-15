# --- pull official base image --- #
ARG BASEIMAGE=python:3.11-slim

FROM $BASEIMAGE as base

ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV JUPYTER_PORT 8989

# ------ Set up workspace -------- #
RUN true && apt-get update
RUN apt update
RUN apt-get install -y curl
RUN apt-get update
RUN apt-get install -y git
RUN apt-get install -y gcc


## --- Install python --- #
RUN true \
    && python -m pip install --upgrade pip \
    && python -m pip install --upgrade pip setuptools \
    && python -m pip install \
      ipython==8.20.0 \
      numpy==1.26.4 \
      pandas==2.1.4 \
      scikit-learn==1.2.2 \
      catboost==1.2 \
      jupyter==1.0 \
      jupyterlab==4.0.8

# --- pull official base image --- #
RUN mkdir /workspace

# --- Make temp file for source code --- #
COPY ./src /workspace/src
COPY ./data /workspace/data
COPY ./notebooks /workspace/notebooks

# --- Set Workspace --- #
WORKDIR "/workspace"

EXPOSE 8989

# --- Start Jupyter Notebook --- #
CMD ["jupyter", "notebook", "--ip", "0.0.0.0", "--no-browser", "--allow-root"]