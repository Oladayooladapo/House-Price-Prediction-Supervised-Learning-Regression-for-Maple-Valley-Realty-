# House-Price-Prediction-Supervised-Learning-Regression-for-Maple-Valley-Realty-
Maple Valley Realty, a forward-thinking real estate agency operating in the Pacific Northwest. The model aims to enhance the company's decision-making by leveraging historical sales data and providing reliable price predictions.

## Project Overview

This project focuses on developing a robust predictive model to estimate residential property prices for **Maple Valley Realty**, a forward-thinking real estate agency. By leveraging historical sales data and property features, the model aims to provide transparent, data-driven insights to buyers, sellers, and investors, enhancing their decision-making process in the real estate market.

This repository contains the Jupyter Notebook detailing the entire data science workflow, from comprehensive Exploratory Data Analysis (EDA) and rigorous data preprocessing to linear regression model development, evaluation, and the derivation of actionable business insights.

## Business Problem

Maple Valley Realty seeks to enhance its data-driven approach to improve customer service and streamline operations. A key challenge is accurately estimating house prices, which is crucial for:
- Advising sellers on optimal pricing strategies.
- Providing buyers with fair market value estimates.
- Assisting investors in identifying potentially undervalued properties.

Without a reliable predictive model, these processes are often reliant on less precise methods, leading to potential inefficiencies and missed opportunities.

## Project Goal

To build, evaluate, and interpret a supervised regression model capable of accurately predicting house prices based on various property features and market data. The project also aims to extract and present actionable business insights that empower Maple Valley Realty's clients.

## Dataset

The dataset used for this project is `house_price_prediction_dataset.csv`, containing various attributes related to residential properties.

**Key Columns:**
- `PropertyID`: Unique identifier for each property. (Dropped during preprocessing)
- `Location`: Neighborhood or area where the property is located.
- `PropertyType`: Type of property (e.g., Single Family, Condo, Townhouse).
- `SizeInSqFt`: Total area of the property in square feet.
- `Bedrooms`: Number of bedrooms in the property.
- `Bathrooms`: Number of bathrooms in the property.
- `YearBuilt`: Year when the property was constructed.
- `GarageSpaces`: Number of garage spaces available.
- `LotSize`: Lot size in square feet.
- `NearbySchools`: Average rating of nearby schools (1-10 scale).
- `MarketTrend`: Recent market trend indicator for the area (1 for increasing, 0 for stable, -1 for decreasing).
- `SellingPrice`: **Target variable**: Actual selling price of the property.

## Project Workflow & Thought Process

My approach to this regression project followed a systematic data science methodology, emphasizing data quality, thorough exploration, and interpretable model building.

### 1. Data Understanding & Initial Inspection
- **Objective:** Get a comprehensive overview of the dataset's structure, content, and initial quality.
- **Steps:**
    - Loaded essential libraries: `pandas`, `matplotlib.pyplot`, `seaborn`.
    - Loaded the `house_price_prediction_dataset.csv` dataset.
    - Used `data.head()` to inspect the first few rows and understand column content.
    - Employed `data.info()` to check data types and identify non-null counts, revealing missing values in `SizeInSqFt`, `Bedrooms`, `Bathrooms`, and `LotSize`.
    - Utilized `data.describe()` to obtain descriptive statistics for numerical columns, observing ranges, means, and standard deviations.
- **Thought Process:** Early data inspection is crucial for understanding the scope of the problem and identifying immediate data quality issues that need addressing.

### 2. Data Cleaning & Preprocessing
- **Objective:** Prepare the raw data for modeling by handling missing values, irrelevant features, and transforming variables into a suitable format.
- **Steps:**
    - **Remove Irrelevant Features:** Dropped `PropertyID` as it's a unique identifier and holds no predictive value for price.
    - **Handle Missing Values:**
        - Identified columns with missing values: `SizeInSqFt`, `Bedrooms`, `Bathrooms`, `LotSize` (each with 5% missing).
        - **Imputation Strategy:** Imputed missing numerical values with the **median** of their respective columns. The median was chosen over the mean to be more robust to potential outliers in these features, ensuring that extreme values do not unduly influence the imputed values.
    - **Encode Categorical Variables:**
        - `Location` and `PropertyType` were categorical features. These were converted into numerical formats using `LabelEncoder`. While One-Hot Encoding is often preferred for nominal categories, Label Encoding was used here. (A note could be added about considering One-Hot for future work if model performance indicates a need for it).
- **Thought Process:** Robust imputation is key for numerical features, especially for sensitive data like property attributes. Encoding categorical data is essential for machine learning models to process them effectively.

### 3. Exploratory Data Analysis (EDA)
- **Objective:** Uncover significant trends, anomalies, and relationships within the data that could influence house prices.
- **Steps & Key Insights:**
    - **Univariate Analysis:**
        - Visualized the distribution of `SellingPrice` (target variable) to understand its spread and identify any skewness.
        - Analyzed distributions of other numerical features (`SizeInSqFt`, `Bedrooms`, `Bathrooms`, `LotSize`, `YearBuilt`, `NearbySchools`) using histograms and box plots to understand their individual characteristics and identify outliers.
    - **Bivariate Analysis:**
        - Explored relationships between `SellingPrice` and other features.
        - **Key Insights Derived:**
            - **SizeInSqFt:** Showed a strong positive correlation with `SellingPrice`. Larger properties command higher prices.
            - **Bedrooms & Bathrooms:** A higher number of bedrooms and bathrooms generally corresponds to higher selling prices.
            - **NearbySchools:** Properties with higher `NearbySchools` ratings tend to have significantly higher `SellingPrice`, highlighting the importance of school quality in property valuation.
            - **Location & PropertyType:** Prices vary significantly across `Location` and `PropertyType`, indicating these are strong determinants of value.
            - **MarketTrend:** Properties in areas with an increasing `MarketTrend` (1) typically sell for more.
    - **Multivariate Analysis (Correlation Heatmap):**
        - Generated a correlation matrix to visualize the relationships between all numerical features.
        - Confirmed strong positive correlations between `SellingPrice` and `SizeInSqFt`, `Bedrooms`, `Bathrooms`, and `LotSize`.
        - Identified other inter-feature correlations (e.g., `YearBuilt` and `NearbySchools` might have indirect relationships with price).
        - **SizeInSqFt** and **Location** are highly correlated with selling price
        - Properties with higher **NearbySchools** ratings show price premium
        - **GarageSpaces** and **LotSize** also contribute positively to price
- **Thought Process:** EDA is an iterative process of questioning the data and visualizing answers. Focusing on the target variable's relationship with other features is paramount for a regression task.

### 4. Model Development & Evaluation
- **Objective:** Build a regression model to predict house prices and rigorously evaluate its performance.
- **Steps:**
    - **Feature and Target Split:** Divided the preprocessed data into independent variables (features, `X`) and the dependent variable (target, `y` - `SellingPrice`).
    - **Data Splitting:** Split the dataset into training and testing sets (e.g., 80% training, 20% testing) to evaluate the model's generalization ability on unseen data.
    - **Model Selection & Training:**
        - A **Linear Regression** model was chosen for its interpretability and as a strong baseline for predicting continuous values.
        - The model was trained on the preprocessed training data.
    - **Model Evaluation:** Assessed the model's performance using key regression metrics:
        - **Mean Absolute Error (MAE):** Average absolute difference between predicted and actual prices (measures average error magnitude).
        - **Mean Squared Error (MSE):** Average of the squared differences (penalizes larger errors more).
        - **Root Mean Squared Error (RMSE):** Square root of MSE (interpretable in the same units as the target variable).
        - **R-squared ($R^2$):** Proportion of the variance in the dependent variable that is predictable from the independent variables (measures how well the model fits the data).
    - **Feature Importance/Coefficients:** Analyzed the coefficients of the linear regression model to understand the impact of each feature on `SellingPrice`.
        - **Summary of Overall Insights:**
            - Larger properties (`SizeInSqFt`) have the highest positive impact on house prices.
            - The number of bedrooms (`Bedrooms`) and bathrooms (`Bathrooms`) significantly increase prices.
            - High-rated schools (`NearbySchools`) are a key driver of pricing in family-friendly neighborhoods.
            - `MarketTrend` also significantly influences price.
- **Thought Process:** Linear Regression is an excellent starting point for regression tasks due to its simplicity and interpretability. Evaluating with multiple metrics provides a holistic view of model performance. Interpreting coefficients directly links the model back to business insights.


### Model Development
- Applied **Multiple Linear Regression**
- Trained/test split with cross-validation
- Evaluated with **R² Score** and **Mean Absolute Error (MAE)**

### Model Performance

| Metric | Value |
|--------|-------|
| R² (Test Set) | ~0.82 |
| MAE | ~$32,000 |

> *Interpretation:* The model explains ~82% of the variance in house prices and predicts the price with an average absolute error of ~$32,000 – acceptable for regional housing markets.

### 5. Strategic Business Recommendations
- **Objective:** Translate the derived data insights and model findings into practical, actionable strategies for Maple Valley Realty.
- **Key Recommendations:**

    1.  **Premium Pricing for School Ratings:**
        * **Insight:** Properties near high-rated schools command significantly higher prices.
        * **Action:** Advise sellers in these areas to price competitively at the higher end. For buyers, highlight school ratings as a key value proposition. Maple Valley Realty could invest in marketing homes specifically emphasizing school district quality.
    2.  **Garage Spaces Influence Value:**
        * **Insight:** The number of `GarageSpaces` positively impacts `SellingPrice`.
        * **Action:** Encourage sellers to detail garage amenities. For marketing, emphasize properties with ample garage space as a value-add.
    3.  **Market Trend is Key:**
        * **Insight:** `MarketTrend` (increasing, stable, decreasing) is a strong predictor of price.
        * **Action:** In hot markets (`MarketTrend = 1`), listing prices can be adjusted upward with confidence. Conversely, in declining markets, a more agile pricing strategy might be needed. Real-time market trend analysis should be integrated into pricing advice.
    4.  **Focus on Size, Bedrooms, Bathrooms:**
        * **Insight:** `SizeInSqFt`, `Bedrooms`, and `Bathrooms` are primary drivers of price.
        * **Action:** For buyers, prioritize properties matching their size and room requirements. For sellers, highlight these features prominently in listings.

## Future Work

To further enhance the model and provide even deeper insights:

-   **Regularized Models:** Incorporate regularized regression models (e.g., Ridge, Lasso) to improve model generalization and prevent overfitting, especially if more features are added.
-   **Non-Linear Models:** Explore more complex non-linear models like Decision Trees, Random Forests, or XGBoost, which can capture more intricate relationships in the data.
-   **External Data Integration:** Integrate external data sources such as:
    -   Crime rates.
    -   Proximity to public transport links.
    -   Local economic indicators (e.g., unemployment rates, job growth).
-   **Geospatial Analysis:** Incorporate geographical data (e.g., latitude/longitude) to perform more advanced spatial analysis and identify hyper-local market trends.
-   **Time-Series Analysis:** If historical sales data includes a time component, consider time-series models to predict future price movements and seasonality more accurately.

## Tools & Libraries Used

-   **Programming Language:** Python
-   **Data Manipulation:** `pandas`, `numpy`
-   **Data Visualization:** `matplotlib.pyplot`, `seaborn`
-   **Machine Learning:** `scikit-learn` (for preprocessing, linear regression, model evaluation)
-   **Jupyter Notebook:** For interactive analysis and documentation.


## Files in this Repository

-   `Case study - House Price Prediction - Supervised Learning (Regression).ipynb`: The main Jupyter Notebook containing all the code for data loading, cleaning, EDA, preprocessing, model training, and evaluation.
-   `house_price_prediction_dataset.csv`: The raw dataset used for the project.
-   `Case Study_ Supervised learning - Regression.pptx.pdf`: The presentation slides outlining the business problem, project workflow, and key findings.
-   `House Price Prediction.docx`: A detailed document outlining the project overview, learning objectives, business context, and recommendations.
-   `README.md`: This file.

