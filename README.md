# Credit-Card-Utility-Segmentation

Segment credit card users based on their **balance**, **credit limit**, and **purchasing behavior** to visualise the group segmentation by using Clustering method.

##  Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Methodology](#methodology)
- [How to Run](#how-to-run)
- [Results & Insights](#results--insights)
- [Dependencies](#dependencies)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

This project performs **customer segmentation** using unsupervised learning techniques to group credit card users according to their behavior profiles—based on variables like balance, credit limit, and purchase activity.

**Use case**: Banks and financial institutions can leverage these segments to tailor marketing strategies, optimize credit offerings, or enhance customer engagement.

---

## Dataset

Include specifics like:
- Source (e.g., Kaggle or internal dataset)
- Key variables/features
- Basic stats: number of users, missing values, etc.

**Example Feature Descriptions**:
| Feature             | Description                                    |
|---------------------|------------------------------------------------|
| `BALANCE`           | Average monthly balance                        |
| `CREDIT_LIMIT`      | Maximum available credit                       |
| `PURCHASES`         | Total purchases in last period                 |
| `CASH_ADVANCE`      | Amount of cash withdrawn                       |

---

## Methodology

1. **Data Preprocessing**  
   - Handling missing values  
   - Scaling/Normalizing data

2. **Dimensionality Reduction (optional)**  
   - PCA for visualization or clustering performance

3. **Clustering Techniques**  
   - K-Means (with elbow or silhouette method)  
   - Optional: DBSCAN or other clustering algorithms

4. **Cluster Profiling & Visualization**  
   - Use plots (boxplots, histograms, scatter, heatmaps)  
   - Interpret segments (e.g., “High Spend Low Credit Limit”, “Frequent Cash Advance Users”)

---

## How to Run

```bash
git clone https://github.com/dhld27/Credit-Card-Utility-Segmentation.git
cd Credit-Card-Utility-Segmentation
python -m venv venv
source venv/bin/activate  # on Windows: .\venv\Scripts\activate
pip install -r requirements.txt
python eda.py         # or other script for EDA
python clustering.py  # start clustering flows
