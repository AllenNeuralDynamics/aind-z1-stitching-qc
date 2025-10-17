# hash:sha256:6214a0235d478016eed9e85c828152fb7bafda22fb273462d6477af0e4bdc58c
ARG REGISTRY_HOST
FROM $REGISTRY_HOST/codeocean/code-server-python-extensions-pack:code-server4.104.3python3.12.4aind-metadata-mcp0.5.2withDOTFILES_REPOsupport

ARG DEBIAN_FRONTEND=noninteractive
ARG GIT_ASKPASS
COPY git-askpass /

RUN pip install -U --no-cache-dir \
    numpy==2.3.4 \
    pandas==2.3.3
