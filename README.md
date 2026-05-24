# PersonaX AI - Digital Behavior Intelligence Platform

PersonaX AI is a production-style Streamlit dashboard for analyzing human digital behavior data. It uses unsupervised machine learning to discover behavioral personas, detect abnormal usage patterns, and generate easy-to-understand insights from CSV data.

The project is designed for a university AI/ML demo and presents the experience like a modern analytics startup product.

## Project Objective

The goal of PersonaX AI is to analyze digital behavior signals such as screen time, app switching, social media usage, gaming time, study time, sleep hours, and productivity score. The system automatically groups users into behavior personas and highlights abnormal digital habits that may indicate addiction risk, low productivity, or unhealthy usage patterns.

## Key Features

- Upload CSV datasets directly from the Streamlit sidebar
- Automatically generate a realistic synthetic demo dataset when no CSV is uploaded
- Handle missing values with median imputation
- Standardize numerical features using `StandardScaler`
- Select custom behavior features for analysis
- Run multiple clustering algorithms:
  - K-Means
  - DBSCAN
  - Agglomerative Clustering
- Automatically generate human-readable persona labels such as:
  - Deep Focus Users
  - Late-Night Scrollers
  - Balanced Users
  - High Dopamine Switchers
  - Productivity Driven Users
  - Immersive Gamers
- Visualize clusters with dimensionality reduction:
  - PCA
  - t-SNE
  - UMAP
- Detect abnormal users with:
  - Isolation Forest
  - Local Outlier Factor
- Display clustering evaluation metrics:
  - Silhouette Score
  - Davies-Bouldin Index
  - K-Means Inertia
- Generate interactive Plotly charts:
  - 2D behavior maps
  - Cluster distribution charts
  - Correlation matrix
  - Persona intensity heatmap
- Provide an AI insights panel for presentation-ready findings
- Download processed CSV data with clusters, personas, and anomaly scores
- Modern dark-themed UI with animated metric cards and responsive layout

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- Plotly
- Pandas
- NumPy
- UMAP-learn

## Project Structure

```text
PersonaX AI - Digital Behavior Intelligence Platform/
├── app.py
├── preprocessing.py
├── clustering.py
├── anomaly_detection.py
├── visualization.py
├── utils.py
├── requirements.txt
├── .gitignore
└── README.md
```

## File Descriptions

| File | Purpose |
| --- | --- |
| `app.py` | Main Streamlit dashboard and UI layout |
| `preprocessing.py` | Synthetic data generation, missing value handling, feature scaling |
| `clustering.py` | K-Means, DBSCAN, Agglomerative Clustering, persona labeling |
| `anomaly_detection.py` | Isolation Forest and Local Outlier Factor anomaly detection |
| `visualization.py` | PCA, t-SNE, UMAP, and Plotly chart functions |
| `utils.py` | Insight generation, formatting helpers, CSV export |
| `requirements.txt` | Python package dependencies |

## Installation

Clone the repository:

```bash
git clone https://github.com/Adarshbandaru/PersonaX-AI-Digital-Behavior-Intelligence-Platform.git
cd PersonaX-AI-Digital-Behavior-Intelligence-Platform
```

Create and activate a virtual environment:

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS/Linux:

```bash
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

## Run the App

```bash
streamlit run app.py
```

If `streamlit` is not recognized, run:

```bash
python -m streamlit run app.py
```

Then open the local URL shown in the terminal, usually:

```text
http://localhost:8501
```

## Expected CSV Format

The app works best with a CSV containing behavior-related numeric columns. Recommended columns:

```text
user_id
screen_time
unlock_frequency
app_switches
social_media_time
gaming_time
study_time
sleep_hours
productivity_score
```

If `user_id` is missing, the app automatically creates user IDs.

## Demo Workflow

1. Start the app with `streamlit run app.py`.
2. Use the built-in synthetic dataset, or upload your own CSV.
3. Select behavior features from the sidebar.
4. Choose a clustering algorithm.
5. Choose a dimensionality reduction method.
6. Choose an anomaly detection model.
7. Explore the dashboard tabs:
   - `Behavior Map`
   - `Personas`
   - `Anomalies`
   - `Data Lab`
   - `Insights`

## Example Presentation Explanation

PersonaX AI takes raw digital behavior data, preprocesses it, clusters users into behavioral personas, detects abnormal usage patterns, and explains the results through interactive visualizations and AI-style insights. The system helps identify groups such as productive users, balanced users, late-night scrollers, and high-switching users.

## Machine Learning Methods Used

### Clustering

The dashboard applies unsupervised clustering to discover natural groups in user behavior data. K-Means is useful for clear persona segmentation, DBSCAN can identify dense groups and noise points, and Agglomerative Clustering provides hierarchy-based grouping.

### Dimensionality Reduction

PCA, t-SNE, and UMAP convert high-dimensional behavior features into 2D visual maps. This makes it easier to understand how users group together.

### Anomaly Detection

Isolation Forest and Local Outlier Factor detect users whose digital behavior is unusual compared with the rest of the dataset. These users are highlighted with anomaly scores.

## Output

The processed dataset includes:

- Original user behavior features
- Cluster ID
- Persona label
- Anomaly score
- Anomaly flag

The processed data can be downloaded as a CSV from the dashboard.

## Author

Developed by Adarsh Bandaru as an AI/ML digital behavior intelligence project.
