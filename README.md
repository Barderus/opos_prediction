# Opossum Intake Forecasting

This project is my attempt to forecast opossum intake volume using historical wildlife rehab intake data. The main focus is predicting daily arrivals for younger opossums, since that is the part of the season that tends to get busy fast and matters most for planning.

## What this project is

At a high level, this repo walks through the full workflow:

- exploring and cleaning the intake data
- building daily arrival time series
- testing baseline, statistical, supervised, and LSTM-style models
- comparing model performance and visualizing the forecast

Most of the work lives in notebooks because I wanted this to feel exploratory and transparent, not hidden behind a giant script.

## Why I did it

I wanted a project that felt applied, useful, and a little more real than a generic forecasting demo. Wildlife rehab data is messy, seasonal, and tied to actual operational questions, which made it a good way to practice data cleaning, feature engineering, model comparison, and communicating results.

Also, opossum baby season is a very real thing, so this was a fun dataset to take seriously without making the project feel overly corporate.

## Goal

The goal was to build a forecasting workflow that could help answer a simple question:

How many young opossums might show up, and when does that seasonal surge start to ramp up?

That meant focusing less on perfect theory and more on building something interpretable, testable, and useful enough to compare approaches side by side.

## Project flow

- `01 - EDA.ipynb`: first pass through the raw intake data
- `02 - Preprocessing.ipynb`: cleaning, filtering, and feature setup
- `03 - Statistical_Modeling.ipynb`: baseline/statistical forecasting work
- `04 - Supervised_Modeling.ipynb`: regression-based models with engineered features
- `05 - Evaluation.ipynb`: model comparison
- `06 - LSTM.ipynb`: sequence modeling experiment
- `07 - Improvement.ipynb`: follow-up tuning and iteration
- `08 - Visualizations.ipynb`: final charts and presentation-ready plots
- `notebooks/forecasting_utils.py`: shared helper functions used across notebooks
