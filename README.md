# Customer Type & Behaviour Analysis on Advertisement

To analyze customer behavior and characteristics in response to online advertisements, identifying key factors that influence ad clicks and providing actionable business recommendations for targeted advertising strategies.

## 📊 Project Overview

This project performs comprehensive customer segmentation and behavior analysis using advertisement interaction data. The analysis explores demographic factors, browsing patterns, and regional characteristics to understand what drives users to click on ads, enabling data-driven marketing decisions.


## 📁 Dataset

Source: Clicked Ads Dataset.csv

Size: 1,000 records with 11 features

Target: Binary classification - whether a user clicked on an ad ("Yes"/"No")

Features Include:

- Daily Time Spent on Site

- Age

- Area Income

- Daily Internet Usage

- Gender

- Geographic data (city, province)

- Advertisement category

- Timestamp

## 📁 Tools & Libraries

Python 3

Libraries:

pandas (v2.2.2)

numpy (v2.0.2)

seaborn (v0.13.2)

matplotlib

## 📁 Repository Structure
```
customer_behavior_project/
│
├── Customer_Type_and_Behaviour_Analysis_on_Advertisement.ipynb  # Main analysis notebook
├── analysis_script.py                                           # Python script version of the notebook
├── requirements.txt                                             # Required dependencies
└── README.md                                                    # Project documentation
```

## 🚀 How to Run the Project

1. Clone the repository:
   git clone https://github.com/nidafebiana/Predict-Customer-Clicked-Ads-Classification-by-Using-Machine-Learning.git

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

3. Run the notebook:
   ```bash
   jupyter notebook Customer_Type_and_Behaviour_Analysis_on_Advertisement.ipynb
   ```

## 🧠 Result & Key Insights 

The analysis reveals several key patterns in customer behavior:

- Age Distribution: Younger users (20-40 years) show different clicking patterns compared to older demographics
- Internet Usage: Correlation between daily internet usage and ad engagement
- Time Spent: Relationship between time spent on site and likelihood of clicking ads
- Geographic Patterns: Regional variations in ad responsiveness
- Income Levels: Impact of area income on advertisement interaction

## 🧠 Business Recommendations & Impact

Key Performance Improvement:

With the same advertising budget (targeting 102 users), the machine learning model increased conversion rate from ~52% → ~96%, generating additional profit of Rp 2.25 million without increasing advertising costs.

Efficiency Gains:

- Machine learning reduced cost per conversion by nearly 2×
- Campaign ROI increased significantly
- Target audience optimization improved ad effectiveness by 84%

Quantified Business Impact:

- Conversion Rate Improvement: +84% (from 52% to 96%)
- Additional Profit: Rp 2.25 million with same budget
- Cost Efficiency: Cost per conversion decreased by 50%
- ROI Increase: Significant improvement in return on investment

## 👤 Author
Nida Febiana
