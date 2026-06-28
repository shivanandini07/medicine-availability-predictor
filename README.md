# Local Medicine Availability Predictor

AI-powered web application that predicts medicine availability at nearby pharmacies using crowdsourced inventory data, machine learning, and OpenStreetMap location services.

## Features

- **Medicine Search** — Search by generic name or brand alias (Paracetamol, Dolo 650, Calpol, Cetzine, Glycomet, Mox, and more)
- **Location Search** — Enter city and full address; coordinates resolved via Google Maps Geocoding API with OpenStreetMap fallback
- **Nearby Pharmacy Discovery** — Finds pharmacies via Overpass API with dataset fallback
- **Availability Prediction** — Random Forest classifier with probability scores
- **Stock-Out Risk** — Low / Medium / High risk with probability
- **Pharmacy Ranking** — Composite score from availability, distance, history, and trend
- **Interactive Map** — Folium map with user location and color-coded pharmacies
- **Analytics Dashboard** — Metrics, inventory trends, and availability charts

## Installation

```bash
git clone <repository-url>
cd project
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate

pip install -r requirements.txt
```

On first run, the app automatically:

1. Generates `data/medicine_inventory.csv` (5000+ synthetic records) if missing
2. Trains and saves `models/medicine_model.pkl` if missing

You can also pre-generate data and train the model manually:

```bash
python model.py
```

## Usage

```bash
streamlit run app.py
```

If `streamlit` is not on your PATH after install, use:

```bash
python -m streamlit run app.py
```

Open the URL shown in the terminal (typically `http://localhost:8501`).

1. Select a medicine in the sidebar
2. Enter your city and address
3. Adjust search radius if needed
4. Click **Predict Availability**

## Project Structure

```
project/
├── app.py                 # Streamlit UI
├── model.py               # Dataset generation & ML training
├── predictor.py           # Prediction, ranking, risk scoring
├── pharmacy_locator.py    # Geocoding & OSM pharmacy search
├── data/
│   └── medicine_inventory.csv
├── models/
│   └── medicine_model.pkl
├── tests/
│   └── test_predictor.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Architecture

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  Streamlit  │────▶│  pharmacy_locator │────▶│  OSM Nominatim  │
│    app.py   │     │  (geocode + OSM)  │     │  Overpass API   │
└──────┬──────┘     └──────────────────┘     └─────────────────┘
       │
       ▼
┌─────────────┐     ┌──────────────────┐     ┌─────────────────┐
│  predictor  │────▶│     model.py      │────▶│  Random Forest  │
│   .py       │     │  features + train │     │  Classifier     │
└─────────────┘     └──────────────────┘     └─────────────────┘
       │
       ▼
┌─────────────────────────────────────────────────────────────┐
│  Ranked results · Risk scores · Folium map · Analytics     │
└─────────────────────────────────────────────────────────────┘
```

### Machine Learning Pipeline

**Features:**

| Feature | Description |
|---------|-------------|
| `historical_availability` | Past availability rate per pharmacy |
| `inventory_quantity` | Reported stock quantity |
| `pharmacy_frequency` | Number of reports from pharmacy |
| `report_recency_days` | Days since last inventory report |
| `medicine_demand_frequency` | How often medicine appears in data |

**Model:** `RandomForestClassifier` (scikit-learn when available) with `StandardScaler`, saved via Joblib. On systems where scikit-learn native extensions are blocked, a pure NumPy Random Forest fallback is used automatically.

**Ranking formula:** 45% availability + 25% distance + 20% history + 10% inventory trend.

## Testing

```bash
pytest tests/ -v
```

Tests cover dataset loading, model training, predictions, ranking, and pharmacy discovery.

## Deployment

### Streamlit Cloud

1. Push the repository to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect the repo and set **Main file path** to `app.py`
4. Deploy

### Render

1. Create a new **Web Service**
2. Build command: `pip install -r requirements.txt && python model.py`
3. Start command: `streamlit run app.py --server.port=$PORT --server.address=0.0.0.0`
4. Add environment variable `PORT` if required

### Railway

1. Create a new project from GitHub
2. Set start command:

```bash
pip install -r requirements.txt && python model.py && streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

3. Expose the generated port

## Requirements

- Python 3.10+
- Internet connection for geocoding and OSM pharmacy lookup (dataset fallback works offline)
- Set `GOOGLE_MAPS_API_KEY` in the environment to use Google Maps Geocoding API for address resolution.

### Set `GOOGLE_MAPS_API_KEY`

- Windows PowerShell:

```powershell
$env:GOOGLE_MAPS_API_KEY = "your_api_key_here"
```

- macOS / Linux:

```bash
export GOOGLE_MAPS_API_KEY="your_api_key_here"
```

- Or create a `.env` file in the project root:

```text
GOOGLE_MAPS_API_KEY=your_api_key_here
```

For permanent environment variables, add the line to your shell profile (for example, `~/.bashrc`, `~/.zshrc`, or PowerShell profile).

## License

MIT
