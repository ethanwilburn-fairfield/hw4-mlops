# hw4-mlops

# Project overview:
This project deploys a machine learning API for predicting customer satisfaction for Olist marketplace orders. The deployed model is a tuned XGBoost pipeline trained on engineered order-level features from the Olist dataset. The goal of the model is to act as an early warning system by predicting whether a customer is likely to leave a positive review (4–5 stars) or a negative review (1–3 stars) based on order, freight, delivery, and customer/order context features before the review is written.

# public URL: https://ethan-hw4-mlops-api.onrender.com

# API documentation:

HEALTH CHECK
Description: Confirms that the API is running and that the model loaded successfully.

GET /health
{
  "status": "healthy",
  "model": "loaded"
}

POST /predict 
Description: Accepts one JSON object containing the required model features and returns:
- binary prediction
- predicted probability
- text label

Request:
{
  "delivery_days": 8.0,
  "delivery_vs_estimated": -15.0,
  "order_hour": 14,
  "order_dayofweek": 2,
  "total_item_price": 278.99,
  "total_freight_value": 28.03,
  "total_order_value": 307.02,
  "freight_ratio": 0.09129698390984302,
  "num_items": 1,
  "num_sellers": 1,
  "order_complexity": 1,
  "is_late": 0,
  "main_product_category": "cool_stuff",
  "main_seller_state": "ES",
  "customer_state": "PR"
}

Response:
{
  "prediction": 1,
  "probability": 0.9064,
  "label": "positive"
}

POST /predict/batch

Description: Accepts a JSON array of up to 100 records and returns predictions for all records.

Request:
{
  "delivery_days": 8.0,
  "delivery_vs_estimated": -15.0,
  "order_hour": 14,
  "order_dayofweek": 2,
  "total_item_price": 278.99,
  "total_freight_value": 28.03,
  "total_order_value": 307.02,
  "freight_ratio": 0.09129698390984302,
  "num_items": 1,
  "num_sellers": 1,
  "order_complexity": 1,
  "is_late": 0,
  "main_product_category": "cool_stuff",
  "main_seller_state": "ES",
  "customer_state": "PR"
},
{
  "delivery_days": 8.0,
  "delivery_vs_estimated": -15.0,
  "order_hour": 14,
  "order_dayofweek": 2,
  "total_item_price": 278.99,
  "total_freight_value": 28.03,
  "total_order_value": 307.02,
  "freight_ratio": 0.09129698390984302,
  "num_items": 1,
  "num_sellers": 1,
  "order_complexity": 1,
  "is_late": 0,
  "main_product_category": "cool_stuff",
  "main_seller_state": "ES",
  "customer_state": "PR"
}

Response:
{
  "count": 2,
  "predictions": [
  {
  "prediction": 1,
  "probability": 0.9064,
  "label": "positive"
},
{
  "prediction": 2,
  "probability": 0.9064,
  "label": "positive"
}

# Input schema:

| Feature               | Data Type            | Valid Values / Range               | Description                                             |
| --------------------- | -------------------- | ---------------------------------- | ------------------------------------------------------- |
| delivery_days         | numeric              | `>= 0`                             | Number of days from purchase to delivery                |
| delivery_vs_estimated | numeric              | any numeric value                  | Difference between actual and estimated delivery timing |
| order_hour            | integer-like numeric | `0` to `23`                        | Hour the order was placed                               |
| order_dayofweek       | integer-like numeric | `0` to `6`                         | Day of week the order was placed                        |
| total_item_price      | numeric              | `>= 0`                             | Total item price for the order                          |
| total_freight_value   | numeric              | `>= 0`                             | Total freight cost for the order                        |
| total_order_value     | numeric              | `>= 0`                             | Total order value                                       |
| freight_ratio         | numeric              | `>= 0`                             | Freight as a share of total order value                 |
| num_items             | integer-like numeric | `>= 0`                             | Number of items in the order                            |
| num_sellers           | integer-like numeric | `>= 0`                             | Number of sellers in the order                          |
| order_complexity      | integer-like numeric | `>= 0`                             | Engineered complexity indicator                         |
| is_late               | integer-like numeric | `0` or `1`                         | Whether the order was delivered late                    |
| main_product_category | string               | must be one of training categories | Main product category                                   |
| main_seller_state     | string               | must be one of training categories | Seller state                                            |
| customer_state        | string               | must be one of training categories | Customer state                                          |

# Local setup:
Option  A - Without Docker

1. Clone or download this repo
2. Make sure the project structure matches
3. Create and activate python environment
4. pip install -r requirements.txt
5. Run the api (python app.py)
6. In a separate terminal test it (python test_api.py)

Option B - With Docker

1. Make sure Docker is installed and running
2. Build docker image:
   docker build -t hw4-api .
3. Run it:
   docker run -p 5000:5000 hw4-api
4. In a seprate terminal test it (python test_api.py)

# Model info:

Deployed model: Tuned XGBoost pipeline (xgb_tuned_pipe)

Prediction target: Positive review (4-5 stars) --> 1
                   Negative review (1-3 stars) --> 0
                   
Key performance metrics:
| Metric    | Foundation Model (review text) | HW2 Model (order features) |
| --------- | -----------------------------: | -------------------------: |
| Accuracy  |                          0.832 |                      0.756 |
| Precision |                          0.955 |                      0.739 |
| Recall    |                          0.778 |                      0.966 |
| F1 Score  |                          0.858 |                      0.837 |

# Known limitations:
- The model was trained on historical structured order data and may degrade as production conditions change.
- The API only supports the feature schema used by the trained model.
- Categorical values must match known training categories.
- The live service may respond slowly on the first request after inactivity because of free-tier hosting behavior.
- The model predicts likely satisfaction before a review is written, so it should not be treated as a direct text sentiment classifier.
