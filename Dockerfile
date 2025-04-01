FROM python:3.12-slim

# 1. Instalación mínima de compiladores
RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    cmake \
    libboost-all-dev \
    #cython3 \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

RUN pip install numpy matplotlib pandas cython jupyter scipy tqdm svgutils

# 2. Crear carpeta de trabajo
WORKDIR /workspace

# 3. Copia del código fuente
COPY . .

# 4. Compilación del simulador
WORKDIR /workspace/edlut/compiled
RUN cmake .. && make -j8

# 5. Crear el symlink para importar desde notebooks
WORKDIR /workspace
RUN ln -s /workspace/edlut/compiled/python /workspace/notebooks/edlut_python_2_rules

# 6. Dejar el contenedor en la raíz del proyecto (útil para ejecutar Jupyter luego)
WORKDIR /workspace/

# 7. Ejecutamos Jupyter
CMD ["jupyter", "notebook", "/workspace/notebooks/main.ipynb", "--ip=0.0.0.0", "--port=8888", "--no-browser", "--allow-root", "--NotebookApp.token=''", "--NotebookApp.password=''"]

