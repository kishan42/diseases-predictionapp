from flask import Flask,request,jsonify,json
import numpy as np
import joblib

diamodel = joblib.load('models/filename.joblib')
heartmodel = joblib.load('models/heartmodel.joblib')
breastmodel = joblib.load('models/breastmodel.joblib')
kedneymodel = joblib.load('models/kedneymodel.joblib')
livermodel = joblib.load('models/livermodel.joblib')

app = Flask(__name__)


@app.route('/')
def home():
    return "hello world !"

@app.route('/diapredict',methods=['POST'])
def diapredict():
    Pregnancies = request.form.get('Pregnancies')
    Glucose = request.form.get('Glucose')
    BloodPressure = request.form.get('BloodPressure')
    SkinThickness = request.form.get('SkinThickness')
    Insulin = request.form.get('Insulin')
    BMI = request.form.get('BMI')
    DiabetesPedigreeFunction = request.form.get('DiabetesPedigreeFunction')
    Age = request.form.get('Age')

    inputq = np.array([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
    result = diamodel.predict(inputq)[0]
    #print(inputq)

    return jsonify({"diabetes":str(result)})

@app.route('/heartpredict',methods=['POST'])
def heartpredict():
    data = request.get_json()

    age = data['age']
    sex = data['sex']
    cp = data['cp']
    trestbps = data['trestbps']
    chol = data['chol']
    fbs = data['fbs']
    restecg = data['restecg']
    thalach = data['thalach']
    exang = data['exang']
    oldpeak = data['oldpeak']
    slope = data['slope']
    ca = data['ca']
    thal = data['thal']


    inputh = np.array([[age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]])
    resulth = heartmodel.predict(inputh)[0]
    
    #print(inputh)

    return jsonify({"heart disease":str(resulth)})

@app.route('/breastpredict',methods=['POST'])
def breastpredict():
    data = request.get_json()

    radius_mean = data['radius_mean']
    texture_mean = data['texture_mean']
    perimeter_mean = data['perimeter_mean']
    area_mean = data['area_mean']
    smoothness_mean = data['smoothness_mean']
    compactness_mean = data['compactness_mean']
    concavity_mean = data['concavity_mean']
    concave_points_mean = data['concave_points_mean']
    symmetry_mean = data['symmetry_mean']
    radius_se = data['radius_se']
    perimeter_se = data['perimeter_se']
    area_se = data['area_se']
    compactness_se = data['compactness_se']
    concavity_se = data['concavity_se']
    concave_points_se = data['concave_points_se']
    fractal_dimension_se = data['fractal_dimension_se']
    radius_worst = data['radius_worst']
    texture_worst = data['texture_worst']
    perimeter_worst = data['perimeter_worst']
    area_worst = data['area_worst']
    smoothness_worst = data['smoothness_worst']
    compactness_worst = data['compactness_worst']
    concavity_worst = data['concavity_worst']
    concave_points_worst = data['concave_points_worst']
    symmetry_worst = data['symmetry_worst']
    fractal_dimension_worst = data['fractal_dimension_worst']




    inputb = np.array([[radius_mean,texture_mean,perimeter_mean,
        area_mean,smoothness_mean,compactness_mean,concavity_mean,
        concave_points_mean,symmetry_mean,radius_se,perimeter_se,
        area_se,compactness_se,concavity_se,concave_points_se,
        fractal_dimension_se,radius_worst,texture_worst,
        perimeter_worst,area_worst,smoothness_worst,
        compactness_worst,concavity_worst,concave_points_worst,
        symmetry_worst,fractal_dimension_worst]])

    resultb = breastmodel.predict(inputb)[0]
    
    #print(inputb)
    
    if(resultb == 0):
        str = "Benign"
    else:
        str = "Malignant"


    return jsonify({"Breast Cancer":str})


@app.route('/kedneypredict',methods=['POST'])
def kedneypredict():
    data = request.get_json()

    age = data['age']
    bp = data['bp']
    al = data['al']
    su = data['su']
    rbc = data['rbc']
    pc = data['pc']
    pcc = data['pcc']
    ba = data['ba']
    bgr = data['bgr']
    bu = data['bu']
    perimeter_se = data['perimeter_se']
    sc = data['sc']
    pot = data['pot']
    wc = data['wc']
    htn = data['htn']
    dm = data['dm']
    cad = data['cad']
    pe = data['pe']
    ane = data['ane']
    

    inputk = np.array([[age,bp,al,su,rbc,pc,pcc,ba,bgr,bu,perimeter_se,sc,pot,wc,htn,dm,cad,pe,ane]])

    resultk = kedneymodel.predict(inputk)[0]

    return jsonify({"Liver disease:":str(resultk)})
    
    #print(inputk)

@app.route('/liverpredict',methods=['POST'])
def liverpredict():
    data = request.get_json()

    Age = data['Age']
    Total_Bilirubin = data['Total_Bilirubin']
    Direct_Bilirubin = data['Direct_Bilirubin']
    Alkaline_Phosphotase = data['Alkaline_Phosphotase']
    Alamine_Aminotransferase = data['Alamine_Aminotransferase']
    Aspartate_Aminotransferase = data['Aspartate_Aminotransferase']
    Total_Protiens = data['Total_Protiens']
    Albumin = data['Albumin']
    Albumin_and_Globulin_Ratio = data['Albumin_and_Globulin_Ratio']
    Gender_Male = data['Gender_Male']
    

    inputl = np.array([[Age,Total_Bilirubin,Direct_Bilirubin,Alkaline_Phosphotase,Alamine_Aminotransferase,Aspartate_Aminotransferase,Total_Protiens,Albumin,Albumin_and_Globulin_Ratio,Gender_Male]])

    resultl = livermodel.predict(inputl)[0]
    
    #print(inputl)   

    return jsonify({"Liver disease:":str(resultl)})


if __name__ == '__main__':
    app.run(host="0.0.0.0",port=8080,debug=True)

