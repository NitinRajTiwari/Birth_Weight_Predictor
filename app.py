from flask import Flask,render_template,request, jsonify
import pickle
import pandas as pd
import traceback
app=Flask(__name__)
import warnings
warnings.filterwarnings('ignore')

@app.route('/')
def home():
    return render_template('index.html')

try:
    with open('model.pkl', mode='rb') as f:
        model = pickle.load(f)
    with open('bmiScaler.pkl', mode='rb') as f:
        bmi_transformer = pickle.load(f)
    with open('gestationScaler.pkl', mode='rb') as f:
        gestation_transformer = pickle.load(f)
    print("Models and transformers loaded successfully!")
except FileNotFoundError as e:
    print(f"Error loading model files: {e}")
    # Set models to None so we can handle the error in the route
    model = None 
    bmi_transformer = None
    gestation_transformer = None
except Exception as e:
    print(f"An unexpected error occurred during model loading: {e}")
    model = None
    bmi_transformer = None
    gestation_transformer = None


@app.route('/predict', methods=["POST"])
def predict():

    if not all([model, bmi_transformer, gestation_transformer]):
        return jsonify({"error": "Models are not available. Check server logs for details."}), 500
    try:
        data = request.get_json()
        d={
            'gestation':data["gestation"],
            'parity':data["parity"],
            "smoke":data["smoke"],
            "BMI":data["BMI"]
        }
        test_df = pd.DataFrame(d)
        test_df["BMI"] = bmi_transformer.transform(test_df[["BMI"]])
        test_df["gestation"] = gestation_transformer.transform(test_df[["gestation"]])
        
        prediction = model.predict(test_df)
        prediction_list = prediction.tolist()
        return jsonify({"Birth Weight":round(prediction_list[0],3)}),200


    except KeyError as e:
        return jsonify({"error": f"Missing column in input data: {str(e)}"}), 400
    except Exception as e:
        print(traceback.format_exc())
        return jsonify({"error": "An unexpected error occurred during prediction."}), 500



if __name__=='__main__':
    app.run(debug=True)