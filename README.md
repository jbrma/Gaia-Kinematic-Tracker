# 🌟 Kinematic Tracing of Runaway Stars using Gaia DR3
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 📖 Abstract
This repository contains an automated, open-source astrophysical pipeline designed to search for surviving binary companions of neutron stars that were ejected after an asymmetric supernova explosion (runaway and walkaway stars). 

Based on kinematic and photometric data from the **Gaia DR3 catalogue**, this tool implements backward tracking algorithms utilizing **Monte Carlo simulations** to evaluate spatial intersections in 3D over hundreds of thousands of years.

## 📁 Project Structure & Data Management
The repository is structured around the main execution folder `pulsars/`, which contains the logic scripts, catalogue processing tools, and a nested storage system indexed by astronomical object:

```text
Gaia-Kinematic-Tracker/
├── README.md
├── requirements.txt
└── pulsars/                            # Main source code directory
    ├── ATNF_Pulsar.ipynb               # Jupyter Notebook for ATNF catalog exploration
    ├── ATNF_PulsarCatalogue.csv        # Raw database of known pulsars
    ├── convert_coords.py               # Utility script for coordinate transformations
    ├── pulsar_targets.csv              # Target list containing physical parameters for multiple pulsars
    ├── run_main.py                     # Batch processing script to run the pipeline sequentially
    ├── main_oop.py                     # Main Object-Oriented pipeline (Core)
    ├── download_gaia.py                # Automated Gaia DR3 ADQL query generator
    ├── dust_generate.py                # Interstellar dust profile extraction tool
    └── pulsars/                        # Data storage subfolder by target
        ├── J0205+6449/                 
        │   ├── fits/
        │   │   └── J0205+6449.fits     # Background astronomical FITS image
        │   ├── starsDB/
        │   │   └── J0205+6449.csv      # Gaia background star catalog
        │   └── Av_profile_J0205+6449.txt # Manually integrated extinction profile
        └── B1800-21/                   
            ├── fits/
            │   └── B1800-21.fits
            ├── starsDB/
            │   └── B1800-21.csv
            └── Av_profile_B1800-21.txt
```

## 🚀 Key Features & Architecture
### Phase 1: Data Preparation
* **`download_gaia.py`**: Calculates the dynamic parallax bounds and generates ADQL queries to download the exact search volume from the Gaia Archive.
* **`dust_generate.py`**: Interfaces with 3D interstellar dust maps to extract Visual Extinction ($A_V$) profiles.
* **Catalogue Management**: Tools like `ATNF_Pulsar.ipynb` and `convert_coords.py` assist in filtering and parsing the official ATNF pulsar catalogue to create the target lists.

### Phase 2: Pipeline Core
* **Photometric Filtering**: Applies interstellar extinction corrections to calculate absolute magnitudes ($M_G$) and intrinsic colors ($(BP-RP)_0$), isolating massive O/B-type candidate stars via Hertzsprung-Russell (H-R) diagram cuts.
* **3D Kinematic Model**: 
  * Replaces deterministic geometric crossings with probabilistic volumetric estimations.
  * Injects a background statistical error (~30 km/s) for stars with missing radial velocities to avoid data loss.
  * Constructs 3D covariance matrices and executes the selected number of Monte Carlo clones per star under a **Moving Target model**.
* **Visualization**: Automatically generates HR diagrams with logarithmic probability color bars, 3D orthogonal projections, and kinematic trajectories overlaid on real FITS astronomical images.

## ⚙️ Installation
Clone the repository and install the required dependencies:

```bash
git clone https://github.com/jbrma/Gaia-Kinematic-Tracker.git
cd Gaia-Kinematic-Tracker
pip install -r requirements.txt
```

*Main dependencies include: `numpy`, `pandas`, `astropy`, `matplotlib`, `pygaia`, `dustmaps`, and `dustmaps3d`.*

## 💻 Usage
Ensure the raw data is placed in its corresponding subfolder (`pulsars/pulsars/PULSAR_NAME/`) before executing the pipeline.

### Option A: Single Target Execution
To run the pipeline on a specific pulsar or supernova remnant, configure the target parameters directly in `main_oop.py` and execute:

```bash
cd pulsars
python main_oop.py
```

### Option B: Batch Processing (Multiple Targets)
To systematically analyze a population of supernova remnants, add their astrometric and age parameters to `pulsar_targets.csv` and use the batch execution script:

```bash
cd pulsars
python run_main.py
```

The script will iterate through the CSV, executing the A/B testing (With H-R Filter vs. Without Filters) for each pulsar and outputting the top candidates and intersection probabilities in `Pulsar_Population_Definitive.xlsx`.

## 📊 Scientific Proof of Concept
As a benchmark test, this pipeline successfully replicated the discovery of "Star A" in the Vela remnant (identifying it as a background interloper via photometry) and successfully identified robust O/B-type massive candidates in systems such as **J0248+6021** and **J1809-2332**.

## 📫 Contact & Author
**Jorge Bravo Mateos**
[LinkedIn](https://www.linkedin.com/in/jorge-bravo-mateos/) | [GitHub](https://github.com/jbrma)
