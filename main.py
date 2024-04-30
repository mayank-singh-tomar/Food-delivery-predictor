from flask import Flask, render_template, request
from sklearn.impute import SimpleImputer
import geocoder
import geopy
import datetime
import joblib
import psycopg2
import numpy as np  

app = Flask(__name__)

# Load pickel file
model = joblib.load('model.pkl')

# Connect to PostgreSQL
conn = psycopg2.connect(
    host="localhost", database="food", user="postgres", password="Tiger")
cur = conn.cursor()

# Using geocoder to get the location based on IP address
g = geocoder.ip('me')
latitude1 = g.latlng[0]
longitude1 = g.latlng[1]

# Fetch current date
current_date = datetime.date.today()

# Separate day, month, and year
day = current_date.day
month = current_date.month
year = current_date.year

# features table
features = {
    'Delivery_person_Age': 0.0,
    'Delivery_person_Ratings': 0.0,
    'Restaurant_latitude': 0.0,
    'Restaurant_longitude': 0.0,
    'Delivery_location_latitude': latitude1,
    'Delivery_location_longitude': longitude1,
    'Vehicle_condition': 1,
    'multiple_deliveries': 0.0,
    'Time_Orderd_Hour': 0.0,
    'Time_Orderd_Minute': 0.0,
    'Time_Orderd_Second': 0.0,
    'Time_Order_pickedHour': 0,
    'Time_Order_pickedMinute': 0,
    'Time_Order_pickedSecond': 0,
    'Order_Date_Year': year,
    'Order_Date_Month': month,
    'Order_Date_Day': day,
    'Weatherconditions_conditions_Fog': 0,
    'Weatherconditions_conditions_NaN': 0,
    'Weatherconditions_conditions_Sandstorms': 0,
    'Weatherconditions_conditions_Stormy': 0,
    'Weatherconditions_conditions_Sunny': 0,
    'Weatherconditions_conditions_Windy': 0,
    'Road_traffic_density_Jam': 0,
    'Road_traffic_density_Low': 0,
    'Road_traffic_density_Medium': 0,
    'Road_traffic_density_NaN': 0,
    'Type_of_order_Drinks': 0,
    'Type_of_order_Meal': 0,
    'Type_of_order_Snack': 0,
    'Type_of_vehicle_electric_scooter': 0,
    'Type_of_vehicle_motorcycle': 0,
    'Type_of_vehicle_scooter': 0,
    'Festival_No': 0,
    'Festival_Yes': 0,
    'City_NaN': 0,
    'City_Semi-Urban': 0,
    'City_Urban': 0
}

# longitude and latitude information based on address

def coordinates(address):
    user_agent = "google_colab_notebook"
    geolocator = geopy.Nominatim(user_agent=user_agent)
    location = geolocator.geocode(address)
    latitude = location.latitude
    longitude = location.longitude
    return latitude, longitude

# Route for the home page
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/location', methods=['GET', 'POST'])
def location():
    if request.method == 'POST':
        loc = request.form['location']
        latitude, longitude = coordinates(loc)
        features['Restaurant_latitude'] = latitude
        features['Restaurant_longitude'] = longitude
    return render_template('order.html')

@app.route('/dashboard', methods=['POST', 'GET'])
def dashboard():
    if request.method == 'POST':
        name = request.form['name']
        time_ordered = request.form['timeOrdered']
        time_ordered_hour, time_ordered_minute = time_ordered.split(':')
        city_type = request.form['city']
        order_type = request.form['ordertype']
        weather_condition = request.form['weatherConditions']
        traffic_condition = request.form['roadTrafficDensity']

        # setting order time
        features['Time_Orderd_Hour'] = int(time_ordered_hour)
        features['Time_Orderd_Minute'] = int(time_ordered_minute)
        if int(time_ordered_minute) < 55:
            features['Time_Order_pickedHour'] = int(time_ordered_hour)
            features['Time_Order_pickedMinute'] = int(time_ordered_minute) + 5
        else:
            features['Time_Order_pickedHour'] = int(time_ordered_hour) + 1
            features['Time_Order_pickedMinute'] = 5

        # changing input for weather condition
        if weather_condition == 'fog':
            features['Weatherconditions_conditions_Fog'] = 1
        elif weather_condition == 'sunny':
            features['Weatherconditions_conditions_Sunny'] = 1
        elif weather_condition == 'stormy':
            features['Weatherconditions_conditions_Stormy'] = 1
        elif weather_condition == 'windy':
            features['Weatherconditions_conditions_Windy'] = 1
        else:
            features['Weatherconditions_conditions_Sandstorms'] = 1

        # changing inputs for Order Type
        if order_type == 'snacks':
            features['Type_of_order_Snack'] = 1
        elif order_type == 'drinks':
            features['Type_of_order_Drinks'] = 1
        else:
            features['Type_of_order_Meal'] = 1

        # changing inputs for traffic condition
        if traffic_condition == 'low':
            features['Road_traffic_density_Low'] = 1
        elif traffic_condition == 'medium':
            features['Road_traffic_density_Medium'] = 1
        else:
            features['Road_traffic_density_Jam'] = 1

        # changing inputs for city type
        if city_type == 'semi-urban':
            features['City_Semi-Urban'] = 1
        else:
            features['City_Urban'] = 1

        # extracting delivery person details from the database
        cur.execute('select delivery_person_age from deliver_data where name=%s', (name,))
        age = cur.fetchone()[0]
        cur.execute('select delivery_person_ratings from deliver_data where name=%s', (name,))
        rating = cur.fetchone()[0]
        cur.execute('select type_of_vehicle from deliver_data where name=%s', (name,))
        vehicle = cur.fetchone()[0]
        cur.execute('select multiple_deliveries from deliver_data where name=%s', (name,))
        multiple_delivery = cur.fetchone()[0]

        # adding to features
        features['Delivery_person_Age'] = age
        features['Delivery_person_Ratings'] = rating
        features['multiple_deliveries'] = multiple_delivery
        if vehicle == 'scooter':
            features['Type_of_vehicle_scooter'] = 1
        elif vehicle == 'motorcycle':
            features['Type_of_vehicle_motorcycle'] = 1
        else:
            features['Type_of_vehicle_electric_scooter'] = 1

        # fetching all the inputs for model
        lst_inputs = list(features.values())

        test_data = np.array(lst_inputs).reshape(1, -1)
        # Instantiate the imputer with the desired strategy (e.g., mean, median, most_frequent)
        imputer = SimpleImputer(strategy='mean')

        # Fit the imputer to your data and transform it
        X_imputed = imputer.fit_transform(test_data)

        #loading to model
        time=model.predict(X_imputed)

    return render_template('order.html', time=time)


@app.route('/developers')  
def developers():
    return render_template('profile.html')     


    
if __name__ == "__main__":
    app.run(debug=True)