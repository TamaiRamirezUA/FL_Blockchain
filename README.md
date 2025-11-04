# **Federated Learning & Blockchain for Vehicular Event Recognition**  
### *Collaborative Traffic Event Recognition Experiments — Dockerized Environment*

This repository contains the full implementation of the experiments described in the article:  
**“Collaborative Traffic Event Recognition with Federated Learning: A Blockchain-Oriented Framework.”**

It includes:  
- **Centralized training code** (baseline)  
- **Federated Learning experiments** using **Flower (FLwr)**  
- A **Dockerized environment** with GPU support  
- Scripts to run, manage, and connect to containers  
- Configuration files for running multi-client FL experiments  

---

## ✅ **Repository Structure**

```
├── centralised/ # Centralized training experiments
│ ├── entrenamiento.py # Main training script
│ └── ...
├── fl-blockchain/ # Flower Federated Learning project
│ ├── pyproject.toml # FL client/server config & experiment parameters
│ ├── ...
├── Dockerfile # GPU-enabled Docker image
├── run.sh # Script to launch a new Docker container
├── connect.sh # Script to connect to an existing container
└── README.md
```

---

## 🚀 **1. Building the Docker Image**

This environment uses a GPU-enabled Docker configuration.

To build the Docker image, run:

```bash
docker build --build-arg USERNAME=$(whoami)-docker \
             --build-arg USER_UID=$(id -u) \
             -t <image_name>:latest .
```

Replace:
* <image_name> → name of your image (e.g., flblockchain)

---

## 🚀 **2. Launching the Docker Environment**

Use the provided script:
```bash
./run.sh <gpu_device> <container_name> <image_name>
```

Examples:

Use a specific GPU
```bash
./run.sh 4090 mycontainer flblockchain
# In our setup we have both NVIDIA RTX 4090 and NVIDIA TITAN so you should adjust the `run.sh` file to meet your set up 
```

Use all GPUs (recommended for servers with 2+ GPUs)
```bash
./run.sh all mycontainer flblockchain
```

---

## 🔗 **3. Connecting to an Existing Container**

If the container is already running:
```bash
./connect.sh <container_name>
```

Example:
```bash
./connect.sh mycontainer
```

---

## ⚙️ **4. Flower (FLwr) Environment Setup**

After building the Docker image, Flower requires an update to your `~/.bashrc`.

Inside the running container, run:
```bash
echo 'export PATH="$HOME/.local/bin:$PATH"' >> ~/.bashrc
source ~/.bashrc
```

This ensures Flower commands and Python local binaries work correctly.

---

## 🧪 **5. Running Centralized Experiments**

The centralized baseline code lives in the `centralised` directory.

Move to that folder:
```bash
cd centralised
```

Then run the training:
```bash
python3 entrenamiento.py
```
This will execute the centralized CNN model used as a baseline in the article.

---

## 🌐 **6. Running Federated Learning Experiments (Flower + Docker)**

The `fl-blockchain` directory contains the complete Flower project.

Navigate to the folder:
```bash
cd fl-blockchain
```

To run the FL experiments, follow Flower’s official deployment guide: 👉 [Flower Documentation](https://flower.ai/docs/framework/how-to-run-flower-with-deployment-engine.html)

Important notes:

* The number of clients, rounds, FL parameters, and experiment variables are set in:
```bash
fl-blockchain/pyproject.toml
```
* Changing the number of clients in `pyproject.toml` must match the number of worker services you deploy using Flower.
* You may launch clients and server in separate terminals inside the Docker environment (or separate containers, depending on your setup).

---

## 📌 **7. Requirements**

All dependencies are already baked into the Docker image.

Key frameworks include:

* Flower (FLwr)
* TensorFlow 2.19
* Keras
* Docker + NVIDIA Container Toolkit
* Python 3.10
* GPU acceleration (CUDA)

---

## 📚 **8. Reference**

If you use this repository, please cite the accompanying article (replace with the final citation if needed):
```
Collaborative Traffic Event Recognition with Federated Learning:
A Blockchain-Oriented Framework
Tamai Ramirez et al., 2025.
```

---

## 🤝 **Contributions**

Contributions, pull requests, and issue reports are welcome.
Please feel free to extend or adapt the FL experiments.

---

## 🤝 Acknowledgements

This work was supported by:

- 🇪🇸 Spanish Research Agency (AEI), Project Serverless4HPC (PID2023-152804OB-I00)
- 🇪🇺 Generalitat Valenciana, CIACIF/2022/176, co-funded by European Social Fund Plus (ESF+)

---

## 📬 **Contact**

For technical issues or collaboration proposals:

**Tamai Ramírez-Gordillo**

📧 **tamai.ramirez@ua.es**  
🏫 University of Alicante  
🌐 [https://www.ua.es](https://www.ua.es)