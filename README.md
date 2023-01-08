
<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li>
      <a href="#about-the-project">About The Project</a>
    <li>
      <a href="#getting-started">Getting Started</a>
      <ul>
        <li><a href="#prerequisites">Prerequisites</a></li>
        <li><a href="#installation">Installation</a></li>
      </ul>
    </li>
    <li><a href="#contact">Contact</a></li>
  </ol>
</details>


<!-- ABOUT THE PROJECT -->
## About The Project - Dynamic Risk Assessment

This project concerns implementation of a automated MLOps pipeline at the production level.
Upon data ingestion and model training, the model is deployed to production and is continuously monitored for data drift, missing values, etc. This essentially happens automatically and is triggered by a cron job. The model is also continuously evaluated and reported on.
Goal of the project was to identify and assess potential risk of customers who may cancel based on their previous activity with the company. <nbr>

The project is part of the Udacity ML Devops Engineer Nanodegree.

<p align="right">(<a href="#readme-top">back to top</a>)</p>

<!-- GETTING STARTED -->
## Getting Started

In order to get started, it is recommended to go step by step through the necessary files (in order):
1. [```ingestion.py```](ingestion.py) : Data ingestion and data integrity checks
2. [```training.py```](training.py): Model training and model preparation (into pickle format), here using a Logistic Regression model
3. [```scoring.py```](scoring.py): Model scoring and model evaluation, here using F1 Score
4. [```deployment.py```](deployment.py): Model deployment to production
5. [```diagnostics.py```](diagnostics.py): Model diagnostics (e.g., identification of data drift, missing values, data summary statistics etc.)
6. [```reporting.py```](reporting.py): Model reporting (e.g., model performance, model diagnostics, etc.)
7. [```app.py```](app.py): Flask app running each functionality above as an API endpoint

All of these functions are also running in [```full_process.py```](fullprocess.py), which is the main file to run the entire pipeline. Also, there is a ```config.json``` which contains all the necessary folder and filenames for the pipeline. An additional file ```apicalls.py``` is provided to run through the functionalities of the API in the Flask app. 

### Prerequisites

- Some knowledge of Python programming [Python](https://www.python.org/)
- Some knowledge of API development with Flask [Flask](https://flask.palletsprojects.com/en/1.1.x/)
- Some knowledge of Machine Learning Model Training and evaluation with [sklearn](https://scikit-learn.org/stable/)
- Some knowledge with cron jobs [cron](https://en.wikipedia.org/wiki/Cron)

### Installation

To install the necessary packages, install them from the **requirements.txt** file:

```python
pip install -r requirements.txt
```

Note: It is recommended to use a virtual environment for this project.

<!-- USAGE EXAMPLES -->
## Usage

To run the Flask app, run the following command in your terminal:

```python
python app.py
```

To run through the functionalities of the API in the Flask app, you can use the following command:

```python
python apicalls.py
```

To run the entire pipeline automatically, run the following command in your terminal:

```sh
service cron start
crontab -e
```
In vim mode, paste in the content of the ```cronjob.txt``` file. Save and exit. The pipeline specified in ```fullprocess.py``` will now run automatically every 10 minutes

<p align="right">(<a href="#readme-top">back to top</a>)</p>


<!-- CONTACT -->
## Contact

Cao Tri Do - [@linkedin](https://www.linkedin.com/in/caotrido/) - huycaotrido@gmail.com

Project Link: [https://github.com/cowtree/dynamic_risk_assessment_mlops](https://github.com/cowtree/dynamic_risk_assessment_mlops)

<p align="right">(<a href="#readme-top">back to top</a>)</p>
