# STR_DA_ACh: Dopamine and Acetylcholine Modulation in the Striatum

This repository contains the full implementation, configuration, and analysis code for the simulations presented in our study on dopaminergic and cholinergic modulation in striatal circuits. The code is bundled in a self-contained Docker environment that launches a Jupyter Notebook to reproduce all results.

---

## ⚡ Quick Start Options

You can either **(A)** pull a prebuilt Docker image, or **(B)** build the container locally.

---

### 🅰️ Option A: Pull Prebuilt Docker Image

#### 1. Install [Docker](https://www.docker.com/products/docker-desktop)

#### 2. Pull and run the container

```bash
docker pull alvarogonzalezredondo/str_da_ach_model
docker run -p 8888:8888 alvarogonzalezredondo/str_da_ach_model
```

#### 3. Open your browser

Go to: [http://localhost:8888](http://localhost:8888)  
The notebook `/workspace/notebooks/main.ipynb` will open automatically.

---

### 🅱️ Option B: Build the Docker Image Locally

#### 1. Clone this repository

```bash
git clone https://github.com/EduardoRosLab/STR_DA_ACh.git
cd STR_DA_ACh
```

#### 2. Build the Docker image

```bash
docker build -t str_da_ach_model .
```

This will compile all necessary components, including the custom neural simulator.

#### 3. Run the container

```bash
docker run -p 8888:8888 str_da_ach_model
```

Then open: [http://localhost:8888](http://localhost:8888)

---

## 🧪 Reproducibility

- The simulation uses a custom C++ neural simulator (EDLUT), compiled automatically during the Docker build.
- All code dependencies are pre-installed (NumPy, Pandas, Matplotlib, Cython, Boost).
- The notebook `/workspace/notebooks/main.ipynb` reproduces all figures from the paper.

---

## 💡 Platform Notes

### Linux
- Works out of the box.
- If permission errors occur, use `sudo` or add your user to the `docker` group.

### Windows
- Use [Docker Desktop for Windows](https://www.docker.com/products/docker-desktop).
- After running the container, open `http://localhost:8888` in your browser.

### macOS
- Use [Docker Desktop for Mac](https://www.docker.com/products/docker-desktop).
- Usage is identical to Windows.

---

## 📫 Contact

For questions or reproducibility issues, please contact:  
**Álvaro González-Redondo** – alvarogr@ugr.es  
GitHub: [https://github.com/EduardoRosLab/STR_DA_ACh](https://github.com/EduardoRosLab/STR_DA_ACh)

---
