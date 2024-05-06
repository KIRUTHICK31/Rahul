from flask import Flask, request, render_template, redirect, url_for
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
import numpy as np
import pandas as pd
import folium
from geopy.geocoders import Nominatim

app = Flask(__name__)

breast_cancer_dataset = load_breast_cancer()
data_frame = pd.DataFrame(breast_cancer_dataset.data, columns=breast_cancer_dataset.feature_names)
data_frame['label'] = breast_cancer_dataset.target
X = data_frame.drop(columns='label', axis=1)
Y = data_frame['label']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=2)
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_train_std = pd.DataFrame(X_train_std, columns=X.columns)
X_test_std = scaler.transform(X_test)

svm_classifier = SVC(kernel='linear', probability=True)
svm_classifier.fit(X_train_std, Y_train)

def get_city_coordinates(city):
    geolocator = Nominatim(user_agent="cancer_hospitals_locator")
    location = geolocator.geocode(city)
    if location:
        return location.latitude, location.longitude
    else:
        print(f"Error: Could not find coordinates for {city}")
        return None, None

def plot_on_map(hospitals, city):
    city_lat, city_lng = get_city_coordinates(city)
    if city_lat is None or city_lng is None:
        return

    map_center = [city_lat, city_lng]
    map_zoom = 10
    map_object = folium.Map(location=map_center, zoom_start=map_zoom)

    for hospital in hospitals:
        name = hospital["name"]
        lat = hospital["lat"]
        lng = hospital["lng"]
        popup_text = f"{name}<br>City: {city}"
        folium.Marker([lat, lng], popup=popup_text).add_to(map_object)

    map_file_path = f"static/cancer_treatment_hospitals_in_{city}.html"
    map_object.save(map_file_path)
    return map_file_path

@app.route('/')
def home():
    return render_template('test.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(request.form[value]) for value in request.form]
        input_data = np.array(input_features).reshape(1, -1)
        input_data_std = scaler.transform(input_data)
        prediction = svm_classifier.predict(input_data_std)
        probas = svm_classifier.predict_proba(input_data_std)
        
        if prediction[0] == 0:
            result = 'Malignant'
            risk_score = probas[0][0]
            cancer_stage = get_cancer_stage(risk_score)
            hospitals_in_city = [
               
                
                {"name": "City Oncology Institute", "city": "Chennai", "lat": 13.0827, "lng": 80.2707},
                {"name": "City Oncology Institute", "city" : "Madurai" ,"lat": 9.91051, "lng": 78.1158},
                {"name": "Ganga Breast Care Centre","city":"Coimbatore","lat":11.046526151070347,"lng": 76.94748059053167},
                {"name": "American Oncology Institute (Cancer Hospital)","city":"Coimbatore","lat":11.076177275669714, "lng": 77.08892955849011},
                {"name": "Dr P Manivannan medical oncologist/cancer specialist","city":"Coimbatore","lat":11.038438960444966, "lng": 76.95160046338484},
                {"name": "KG Hospital","city":"Coimbatore","lat":11.00083080418529, "lng":76.97145914358717},
                {"name": "Shifa Hospitals","city":"Tirunelveli","lat":8.729605403052377, "lng": 77.71022631451892},
                {"name": "Annai Velankanni Multispeciality Hospital","city":"Tirunelveli","lat":8.727518177996128, "lng": 77.73029204430543},
                {"name": "Dr. Thiru Neuro & Multispeciality Hospital","city":"Salem","lat":11.66277358276269,  "lng": 78.2036057991546},
                {"name": "S P M M Hospital","city":"Salem","lat":11.661495262076214,"lng": 78.19215709753885},
                {"name": "Deepam Speciality Hospital","city":"Salem","lat":11.627225753359461, "lng": 78.14765716785244},
                {"name": "Erode Cancer Centre","city":"Erode","lat":11.314214379688108, "lng": 77.66866851526882},
                {"name": "Care 24 Medical Centre And Hospital","city":"Erode","lat":11.326723666018433, "lng": 77.68815100182704},
                {"name": "Sudha Cancer Centre","city":"Erode","lat":11.339469808925454, "lng": 77.71060816894833},
                {"name": "Dharshini Hospitals","city":"Dindigul","lat":10.369450269228487, "lng": 77.97670400179244},
                {"name": "Silverline Speciality Hospital","city":"Trichy","lat":10.82495705981534, "lng": 78.68191979105109},
                {"name": "Dr G Viswanathan - CBCC cancer center","city":"Trichy","lat":10.854306183909827,"lng": 78.69767755814395},
                {"name": "Muthu Neuro & Trauma Centre","city":"Nagercoil","lat":8.197746599024681,"lng":  77.38772552578038},
                {"name": "Siva Hospitals","city":"Nagercoil","lat":8.122241950842723, "lng": 77.39304911998887},
                {"name": "Thangam Hospital","city":"Namakkal","lat":11.215612293093322,"lng":  78.17210515101631,},
                {"name": "Sarvam Multi System Of Medicine and Research Institute Pvt Ltd","city":"Namakkal","lat":11.241817630447537, "lng":  78.17210515101631},
                {"name": "Sri Narayani Hospital and Research Centre","city":"Vellore","lat":12.870774704836, "lng":  79.08993488909442},
                {"name": "Nalam Medical Centre and Hospital","city":"Vellore","lat":12.935267352497144,"lng": 79.15374655748693},
                {"name": "Fortis Hospital Richmond Road","city":"Bangalore","lat":12.980093019450257,  "lng":  77.61120678099982},
                {"name": "Cytecare Hospitals Bangalore","city":"Bangalore","lat":13.11866973853396,  "lng": 77.60823194897264},
                {"name": "Sri Shankara Cancer Foundation","city":"Bangalore","lat":12.95430507042922,  "lng": 77.5711659383228},
                {"name": "HCG Cancer Hospital - (K. R. Road, Bengaluru)","city":"Bangalore","lat":12.96453572401942,   "lng": 77.58958136405364},
                {"name": "OncoVille Cancer Hospital & Research Centre","city":"Bangalore","lat":12.97643205249809, "lng":77.50963115181774 },
                {"name": "Healius Cancer & Hematology - Cancer Hospitals","city":"Bangalore","lat":12.929307978652597, "lng":77.57335695106764}
                
                
                
            ]
            
            return redirect(url_for('prediction_result', result=result, cancer_stage=cancer_stage))

        else:
            result = 'Benign'
            return redirect(url_for('prediction_result', result=result))
        
    except Exception as e:
        return render_template('test.html', prediction_text=f'Error: {str(e)}')
#map_file_path = plot_on_map(hospitals_in_city, "Madurai") 
# , map_file=map_file_path
def get_cancer_stage(risk_score):
    if risk_score < 0.1:
        return 'Stage 0 (Carcinoma in situ)'
    elif risk_score < 0.2:
        return 'Stage 1A'
    elif risk_score < 0.3:
        return 'Stage 1B'
    elif risk_score < 0.4:
        return 'Stage 2A'
    elif risk_score < 0.5:
        return 'Stage 2B'
    elif risk_score < 0.6:
        return 'Stage 3A'
    elif risk_score < 0.7:
        return 'Stage 3B'
    elif risk_score < 0.8:
        return 'Stage 3C'
    else:
        return 'Stage 4'
@app.route('/prediction_result')
def prediction_result():
    result = request.args.get('result')
    cancer_stage = request.args.get('cancer_stage', None)
    map_file_path = request.args.get('map_file', None)
    if cancer_stage:
        return render_template('prediction_result.html', result=result, cancer_stage=cancer_stage, map_file=map_file_path)
    else:
        return render_template('prediction_result.html', result=result)


if __name__ == '__main__':
    app.run(debug=True)
