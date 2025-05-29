# Secure-Beam-MPA: Proactive mmWave Security

## Secure Beamforming in mmWave Networks: A Multimodal Predictive Approach with Strategic Alignment

This project introduces a cutting-edge framework for enhancing security and efficiency in millimeter-wave (mmWave) wireless networks. It moves beyond traditional reactive defenses by employing a proactive and predictive strategy.

The core of the system is a **multimodal deep learning model** that processes both **Channel State Information (CSI)** and **Integrated Sensing and Communications (ISAC)**-derived threat intelligence. This allows it to simultaneously predict user location and select the optimal communication beam.

A significant innovation is the use of **Direct Preference Optimization (DPO)** to strategically align the beamforming model's decisions with crucial operational policies, such as prioritizing security (e.g., avoiding beams directed towards potential attackers) and maintaining link stability alongside performance.

The simulation is developed in **Python**, leveraging **TensorFlow** for deep learning and the **Sionna** library for realistic link-level simulations, including its 3GPP TR 38.901 channel models and MIMO functionalities.

---

## 🔑 Key Features

- 🚀 **Predictive Dual-Task Model**: Simultaneously forecasts user location and selects the optimal beam.  
- 🔗 **Multimodal Data Fusion**: Integrates CSI with ISAC-based threat intelligence for robust, context-aware decisions.  
- 🛡️ **Strategic Alignment with DPO**: Fine-tunes the model using Direct Preference Optimization to ensure decisions prioritize security and link stability.  
- 📡 **Sionna-based Simulation**: Employs Sionna for accurate mmWave channel modeling and advanced beamforming techniques (DFT-based beam codebooks).  
- ⚙️ **End-to-End Workflow**: Provides a complete pipeline from data generation and model training (pre-training & DPO alignment) to comprehensive evaluation.

---

## 📦 Prerequisites

Ensure you have the following installed:

- Python 3.8+
- TensorFlow 2.10+
- Sionna 0.15+ ([Official Installation Guide](https://nvidia.github.io/sionna/))
- NumPy
- Pandas
- tqdm

You can install the required Python packages using:

```bash
pip install tensorflow numpy pandas tqdm
