# Women Safety Crime Hotspot Prediction System

An AI-powered Streamlit web application for women safety prediction, hotspot analysis, and crime trend visualization using crime data from India.

## Project Structure

- `app.py` - Streamlit application entrypoint
- `train_model.py` - Model training and export script
- `requirements.txt` - Python package dependencies
- `dataset/` - Holds the crime dataset and sample data
- `models/` - Stores trained machine learning artifacts
- `utils/` - Modular preprocessing, modeling, visualization, and helper utilities
- `assets/` - Custom styling and static content

## Features

- Crime hotspot detection and area-wise risk prediction
- Women safety score calculation and severity categorization
- Yearly crime trends and interactive analytics
- Modern professional UI with sidebar navigation and dashboard cards
- Emergency helpline section and safety tips for women
- Automatic model selection between multiple classifiers
- Downloadable prediction report

## Setup Instructions

1. Clone the repository:

```bash
git clone https://github.com/your-username/women-safety-crime-hotspot-prediction.git
cd "Women Safety Prediction"
```

2. Create a Python virtual environment:

```bash
python -m venv venv
venv\Scripts\activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Add the Kaggle datasets:

- Download the "Crimes in India Dataset" from [Kaggle](https://www.kaggle.com/datasets/rajanand/crime-in-india) and place it as `dataset/crime_in_india.csv`.
- Download the "Crime Against Women in India (2001-2014)" dataset from [Kaggle](https://www.kaggle.com/datasets/greeshmagirish/crime-against-women-20012014-india) and place it as `dataset/crime_against_women.csv`.

> If the full Kaggle datasets are not present, the application will still run using the sample dataset in `dataset/sample_crime_data.csv`.

5. Train the model (optional, the app can train automatically if needed):

```bash
python train_model.py
```

6. Run the Streamlit app:

```bash
streamlit run app.py
```

## Deployment to Streamlit Community Cloud

1. Push your repository to GitHub.
2. Go to [Streamlit Community Cloud](https://share.streamlit.io/) and sign in.
3. Create a new app and connect your GitHub repository.
4. Set the main file to `app.py` and deploy.

## Notes

- The app automatically detects if `models/best_crime_model.joblib` exists and loads it.
- If the dataset is not available locally, sample data is used so the UI remains functional.

## Contact

- Developer: Women Safety Prediction System
- GitHub: https://github.com/your-username
- Portfolio: https://www.linkedin.com/in/your-profile
