# Urban Feature Classification Using Satellite Imagery

An end-to-end machine learning and GIS-based system for automatically identifying, classifying, and mapping urban land-cover features from satellite imagery. The project explores classical machine learning algorithms and deep learning techniques, evaluates their classification performance, and generates thematic maps to support urban characterisation and informed planning decisions.

<p align="center">
  <a href="https://satellite-urban-classification.streamlit.app/">
    <strong>🚀 Live Web Application</strong>
  </a>
</p>

---

## 1. Project Overview

Rapid urbanisation has resulted in increasingly complex urban landscapes comprising buildings, roads, vegetation, water bodies, barren land, and other land-cover features. Accurate identification and mapping of these features are essential for urban planning, infrastructure development, environmental monitoring, and sustainable resource management.

This project develops an automated urban land-cover classification pipeline using satellite imagery, GIS-based annotation, machine learning, and deep learning.

The system involves:
- Preparing satellite imagery and labelled training datasets using QGIS.
- Digitising land-cover polygons and generating pixel-level reference labels.
- Extracting spectral and spatial features from satellite imagery.
- Training and evaluating multiple machine learning and deep learning models.
- Generating thematic maps representing the classified urban landscape.
- Developing a web-based application for interactive image classification and model evaluation.

## 2. Objectives

1. Identify relevant urban land-cover categories from satellite imagery.
2. Prepare labelled geospatial datasets through manual polygon annotation.
3. Explore classical machine learning and deep learning approaches for pixel-level classification.
4. Compare the performance of different algorithms using quantitative evaluation metrics.
5. Generate thematic maps to visualise the spatial distribution of urban features.
6. Develop a web-based interface to make the classification pipeline accessible.

## 3. Land-Cover Classes

The project considers five land-cover categories:

| Class | Description |
|---|---|
| Vegetation | Vegetated areas, including green spaces and plant-covered regions |
| Road | Road networks and paved transportation surfaces |
| River | Rivers and other identified water channels |
| Barren Land | Exposed soil and non-vegetated open land |
| Built-up | Buildings and other developed urban surfaces |

The classification labels are represented using numeric class IDs in the annotated geospatial datasets.

## 4. Technology Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| GIS and Geospatial Processing | QGIS, Rasterio, GeoPandas |
| Numerical Computing | NumPy, Pandas |
| Machine Learning | Scikit-learn |
| Deep Learning | TensorFlow / Keras |
| Image Processing | Pillow, Rasterio |
| Visualisation | Matplotlib |
| Web Application | Streamlit |
| Geospatial Annotation | QGIS |
| Version Control | Git, GitHub |

## 5. Methodology

### 5.1 Geospatial Data Preparation

QGIS was used to inspect, prepare, and annotate the satellite imagery.

The data preparation workflow involved:

- Loading the satellite imagery into QGIS.
- Inspecting raster properties and coordinate reference system information.
- Preparing separate training and testing satellite images.
- Digitising polygon geometries corresponding to different land-cover categories.
- Assigning class IDs and class names to the annotated polygons.
- Saving the annotations in GeoPackage (`.gpkg`) format.

Separate training and testing annotations were prepared to support independent model evaluation.

### 5.2 Ground-Truth Generation

The annotated polygons were converted into pixel-level reference labels through rasterisation.

Each pixel covered by an annotated polygon was assigned its corresponding numeric class ID.

These rasterised labels serve as ground truth for supervised model training and evaluation.

### 5.3 Feature Extraction

The classification pipeline extracts spectral and spatial information from satellite imagery.

The extracted information is used to construct feature representations for the classical machine learning algorithms.

The feature extraction process enables the models to distinguish between different land-cover categories based on their image characteristics.

### 5.4 Machine Learning Models

Five approaches were explored:

**1. Logistic Regression**

A supervised classification algorithm used to establish a baseline for multiclass land-cover classification.

**2. Support Vector Machine (SVM)**

A supervised learning approach that identifies decision boundaries between land-cover categories using feature-space separation.

**3. Random Forest**

An ensemble learning algorithm that combines predictions from multiple decision trees to perform classification.

**4. K-Means Clustering**

An unsupervised learning algorithm that groups pixels into clusters based on feature similarity. It provides a clustering-based approach to exploring land-cover patterns.

**5. Convolutional Neural Network (CNN)**

A deep learning approach that learns spatial patterns and image representations through convolutional operations. It is used to explore the effectiveness of spatial context in satellite-image classification.

### 5.5 Model Evaluation

The models are evaluated using quantitative classification metrics, including:

- Overall Accuracy
- Precision
- Recall
- Macro F1-Score
- Confusion Matrix
- Cohen's Kappa Score

The evaluation results are used to compare classification performance and understand class-wise prediction behaviour.

### 5.6 Thematic Map Generation

The trained models generate pixel-level predictions over satellite imagery.

The predicted class IDs are converted into colour-coded thematic maps, allowing the spatial distribution of different urban land-cover categories to be visualised.

The outputs include:
- CNN thematic maps.
- K-Means classification maps.
- Logistic Regression classification maps.
- Random Forest classification maps.
- SVM classification maps.

Both training-region and testing-region visualisations are generated for analysis.

## 6. Project Outputs

The project generates the following outputs:

| Output | Purpose |
|---|---|
| Classification maps | Visual representation of predicted land-cover classes |
| Training maps | Visualisation of predictions over the training imagery |
| Testing maps | Visualisation of predictions over independent test imagery |
| Confusion matrices | Analysis of class-wise prediction performance |
| Classification reports | Precision, recall and F1-score evaluation |
| CNN training logs | Monitoring training and validation performance |
| Model comparison results | Comparison of different classification approaches |

## 7. Web Application

A web-based application has been developed using Streamlit to provide an interactive interface for satellite-image classification.

**Live Application:** https://satellite-urban-classification.streamlit.app/

The application provides functionality for:

- Uploading satellite imagery.
- Selecting a classification model.
- Generating thematic classification maps.
- Visualising predicted land-cover distributions.
- Exploring classification results through an interactive interface.

The application makes the developed classification pipeline accessible without requiring users to execute the underlying Python scripts manually.

## 8. Installation and Setup

### Prerequisites

- Python 3.10 or later (subject to dependency compatibility).
- Git.
- QGIS for geospatial annotation and inspection.

### Clone the Repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_NAME>
```

### Create a Virtual Environment

```bash
python -m venv venv
```

Activate it:

**Windows**
```bash
venv\Scripts\activate
```

**Linux / macOS**
```bash
source venv/bin/activate
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

## 9. Running the Project

### Run the Web Application

```bash
streamlit run app.py
```

Open the local URL displayed in the terminal to access the application.

### Run the Classical Machine Learning Training Pipeline

The training script accepts separate training and testing satellite images and their corresponding GeoPackage annotations.

```bash
python train_classical.py train_satellite.tif train_data.gpkg test_satellite.tif test_data.gpkg
```

Replace the filenames with the paths to your respective datasets.

The pipeline processes the supplied data, trains the supported classical machine learning models, evaluates their performance, and generates classification outputs.

*Refer to the project source code for the CNN training and inference workflow.*

## 10. Repository Structure

The repository is organised around the following major components:

```text
urban-feature-classification/
│
├── app.py
│
├── train_classical.py
│
├── common.py
│
├── requirements.txt
│
├── models/
│   ├── logreg
│   ├── svm
│   ├── random_forest
│   ├── kmeans
│   └── cnn
│
├── data/
│   ├── training/
│   └── testing/
│
├── outputs/
│   ├── classification_maps/
│   ├── confusion_matrices/
│   └── evaluation_results/
│
└── README.md
```

*Note: This is an illustrative logical structure. Adjust the directory names to match the actual repository contents.*

## 11. Results and Analysis

The project investigates the effectiveness of different classification approaches for identifying urban land-cover categories.

The generated thematic maps enable qualitative comparison of:
- Spatial continuity of classified regions.
- Identification of different land-cover patterns.
- Boundary representation between neighbouring classes.
- Differences between supervised and unsupervised classification.
- The ability of deep learning models to utilise spatial information.

Quantitative evaluation metrics and confusion matrices provide additional evidence for comparing model performance.

Actual model scores should be reported alongside their corresponding experimental configurations and test datasets.

## 12. Limitations and Future Scope

Potential areas for further development include:

- Incorporating multispectral satellite imagery and additional spectral indices.
- Improving the quality and spatial coverage of annotated training datasets.
- Exploring advanced CNN architectures and semantic segmentation networks.
- Investigating class imbalance and improving minority-class prediction.
- Performing experiments across multiple geographical regions.
- Incorporating higher-resolution satellite imagery.
- Improving spatial generalisation across different urban environments.
- Adding interactive GIS visualisation and downloadable geospatial outputs to the web application.

## 13. Applications

The developed workflow has potential applications in:

- Urban land-use and land-cover mapping.
- Urban expansion monitoring.
- Infrastructure and road-network identification.
- Vegetation and green-space assessment.
- Water-body mapping.
- Environmental monitoring.
- Sustainable urban planning.

## 14. Acknowledgements

This project was undertaken as part of the CE235 course project at the Indian Institute of Technology Bombay (IIT Bombay).

We acknowledge the role of open-source GIS and machine learning technologies, particularly QGIS, Python, Scikit-learn, TensorFlow, and Streamlit, in enabling the development of the project.

---

**Project:** Classification and Mapping of Urban Features Using Satellite Imagery

**Institution:** Indian Institute of Technology Bombay

**Course:** CE235

**Live Demo:** https://satellite-urban-classification.streamlit.app/
