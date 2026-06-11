# 🍔 Food Delivery Time Predictor

## Overview

Food Delivery Time Predictor is a Machine Learning web application that estimates the expected delivery time of food orders based on various factors related to the delivery person, restaurant location, customer location, weather conditions, traffic density, and order details.

The goal of this project is to help food delivery platforms and customers obtain accurate delivery time predictions, leading to better planning, improved customer satisfaction, and optimized delivery operations.

---

## Features

* Predicts food delivery time in minutes.
* Uses real-world delivery and location-based factors.
* Considers weather and traffic conditions.
* Easy-to-use web interface.
* Data-driven predictions using Machine Learning algorithms.

---

## Dataset Features

The model uses the following input features:

| Feature                     | Description                        |
| --------------------------- | ---------------------------------- |
| ID                          | Unique order identifier            |
| Delivery_person_ID          | Unique delivery partner identifier |
| Delivery_person_Age         | Age of delivery person             |
| Delivery_person_Ratings     | Average rating of delivery person  |
| Restaurant_latitude         | Restaurant latitude coordinate     |
| Restaurant_longitude        | Restaurant longitude coordinate    |
| Delivery_location_latitude  | Customer latitude coordinate       |
| Delivery_location_longitude | Customer longitude coordinate      |
| Order_Date                  | Date of order                      |
| Time_Orderd                 | Time when order was placed         |
| Time_Order_picked           | Time when order was picked up      |
| Weatherconditions           | Weather during delivery            |
| Road_traffic_density        | Traffic density level              |
| Vehicle_condition           | Condition of delivery vehicle      |
| Type_of_order               | Type of food order                 |
| Type_of_vehicle             | Vehicle used for delivery          |
| Multiple_deliveries         | Number of simultaneous deliveries  |
| Festival                    | Festival indicator (Yes/No)        |
| City                        | Delivery city category             |

### Target Variable

| Feature         | Description                     |
| --------------- | ------------------------------- |
| Time_taken(min) | Actual delivery time in minutes |

---

## How It Works

The prediction model analyzes multiple factors that influence delivery duration, including:

* Distance between restaurant and customer
* Traffic conditions
* Weather conditions
* Delivery partner experience and ratings
* Vehicle type and condition
* Number of ongoing deliveries
* City and festival impact
* Order and pickup timings

Using these inputs, the trained machine learning model predicts the estimated delivery time for a new order.

---

## Tech Stack

### Frontend

* HTML
* CSS
* JavaScript

### Backend

* Python
* Flask

### Machine Learning

* Pandas
* NumPy
* Scikit-learn

---

## Project Workflow

1. Data Collection
2. Data Cleaning and Preprocessing
3. Feature Engineering
4. Model Training
5. Model Evaluation
6. Deployment using Flask
7. User Input and Prediction

---

## Installation

```bash
git clone https://github.com/your-username/food-delivery-time-predictor.git

cd food-delivery-time-predictor

pip install -r requirements.txt

python app.py
```

---

## Usage

1. Open the application in your browser.
2. Enter delivery details such as:

   * Delivery Person Age
   * Delivery Person Rating
   * Weather Conditions
   * Traffic Density
   * Vehicle Type
   * Restaurant Location
   * Customer Location
   * City and Festival Information
3. Click **Predict**.
4. The system will display the estimated delivery time in minutes.

---

## Future Improvements

* Real-time traffic integration using Maps API.
* Live weather data integration.
* Deep Learning-based prediction models.
* Route optimization recommendations.
* Mobile application deployment.

---

## Conclusion

This project demonstrates how Machine Learning can be applied to solve real-world logistics and delivery challenges. By analyzing delivery-related factors, the system provides accurate delivery time estimates that can help businesses improve operational efficiency and customer experience.

---

### Author

**Mayank Singh Tomar**

Food Delivery Time Predictor – Machine Learning Project
