FROM jupyter/scipy-notebook


USER root

# Autocompletion
RUN pip install jupyterlab-lsp
RUN pip install 'python-lsp-server[all]'

# Complex networks
RUN pip install igraph
