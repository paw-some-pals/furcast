# FurCast

<p align="center">
  <img src="images/furcast-logo.png" alt="FurCast logo" width="300">
</p>

<p align="center">
  <strong>A forecasting application for animal shelters</strong>
</p>

FurCast is a machine learning project developed by Edmonton Team E1 during the **AI4Good Lab 2026**. 
It predicts an animal’s expected length-of-stay category and likely outcome type, helping shelter staff data-informed decisions about care and resources.

## The Problem

Animal shelters and rescue organizations often operate at or near capacity. Through our stakeholder research, one Alberta rescue reported being able to accept only **5% of rescue requests per day**.

## Our Solution

FurCast supports two classification tasks:

1. **Length-of-stay prediction:** Predicts whether an animal is likely to have a short, medium, or long stay.
2. **Outcome-type prediction:** Predicts one of five possible outcomes and provides suggestions to help staff plan for the animal’s stay.

The system is designed to support shelter staff, not replace their experience or professional judgment.

## Data and Features

We combined shelter records with additional information to create a more complete view of each animal and its environment.

The feature groups included:

* Intake information
* Breed information
* Municipal information
* Economic and demographic information

## Models and Evaluation

### Outcome-Type Classification

A **Random Forest** model was trained separately for cats and dogs.

Our strongest reported result was **86.6% accuracy** for dog length-of-stay prediction, compared with a **60% baseline accuracy**.

## Technologies

* Python
* Pandas
* Scikit-learn
* Random Forest classification
* Data cleaning and feature engineering
* Model evaluation and ensemble learning

## Limitations

Some factors that can strongly affect an animal’s stay were not consistently tracked in the available datasets, including:

* Shelter capacity at the time of intake
* Foster-home availability

## Ethical Considerations

* FurCast is intended to **support shelter staff**, not replace expert judgment.
* Cases involving sick or distressed animals should always be reviewed by an appropriate professional.

