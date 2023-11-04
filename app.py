from flask import Flask, render_template,request, jsonify
import pandas as pd
import pickle
import os
import tensorflow as tf
import numpy as np

app = Flask(__name__)


@app.route('/')
def home():
    return render_template("index.html")
@app.route('/groceries') 
def groceries():
    data = pd.read_csv('models and notebooks/groceries1.csv')
    data['shortened_name'] = data['name'].apply(lambda x: ' '.join(x.split()[:10]))
    items = data.to_dict(orient='records')
    return render_template('jrocieries.html', items=items)

@app.route('/appliances') 
def appliances():
    data = pd.read_csv('models and notebooks/appliances.csv')
    data['shortened_name'] = data['name'].apply(lambda x: ' '.join(x.split()[:10]))
    items = data.to_dict(orient='records')
    return render_template('appliances.html', items=items)

@app.route('/kitchen') 
def kitchen():
    data = pd.read_csv('models and notebooks/HomeandKitchen.csv')
    data['shortened_name'] = data['name'].apply(lambda x: ' '.join(x.split()[:10]))
    items = data.to_dict(orient='records')
    return render_template('kitchen.html', items=items)

@app.route('/fruits') 
def fruits():
    data = pd.read_csv('models and notebooks/Fruitsandveg.csv')
    items = data.to_dict(orient='records')
    return render_template('fruits.html', items=items)

@app.route('/select_groceries/<selected_item>')
def select_item(selected_item):
    data = pd.read_csv('models and notebooks/groceries1.csv')
    data['shortened_name'] = data['name'].apply(lambda x: ' '.join(x.split()[:10]))
    with open('models and notebooks/cosine_similarity_matrix.pkl', 'rb') as file:
        cosine_si = pickle.load(file)
    def recommend(items):
        index = data[data['name'] == items].index[0]
        distances = sorted(list(enumerate(cosine_si[index])), reverse=True, key=lambda x: x[1])
        recommended_items = [data.iloc[i[0]]['name'] for i in distances[1:9]]
        return recommended_items
    recommended_items = recommend(selected_item)
    return render_template('select_groceries.html', selected_item=selected_item, recommended_items=recommended_items,data=data)

@app.route('/select_appliances/<selected_item>')
def select_appliances(selected_item):
    data = pd.read_csv('models and notebooks/appliances.csv')
    data['shortened_name'] = data['name'].apply(lambda x: ' '.join(x.split()[:10]))
    with open('models and notebooks/appliances.pkl', 'rb') as file:
        cosine_si = pickle.load(file)
    def recommend(items):
        index = data[data['name'] == items].index[0]
        distances = sorted(list(enumerate(cosine_si[index])), reverse=True, key=lambda x: x[1])
        recommended_items = [data.iloc[i[0]]['name'] for i in distances[1:9]]
        return recommended_items
    recommended_items = recommend(selected_item)
    return render_template('select_appliances.html', selected_item=selected_item, recommended_items=recommended_items,data=data)

@app.route('/select_kitchen/<selected_item>')
def select_kitchen(selected_item):
    data = pd.read_csv('models and notebooks/HomeandKitchen.csv')
    data['shortened_name'] = data['name'].apply(lambda x: ' '.join(x.split()[:10]))
    with open('models and notebooks/HomeandKitchen.pkl', 'rb') as file:
        cosine_si = pickle.load(file)
    def recommend(items):
        index = data[data['name'] == items].index[0]
        distances = sorted(list(enumerate(cosine_si[index])), reverse=True, key=lambda x: x[1])
        recommended_items = [data.iloc[i[0]]['name'] for i in distances[1:9]]
        return recommended_items
    recommended_items = recommend(selected_item)
    return render_template('select_kitchen.html', selected_item=selected_item, recommended_items=recommended_items,data=data)

@app.route('/select_fruits/<selected_item>')
def select_fruits(selected_item):
    data = pd.read_csv('models and notebooks/Fruitsandveg.csv')
    return render_template('select_fruits.html', selected_item=selected_item,data=data)


# Load the CSV file into a Pandas DataFrame
df = pd.read_csv("models and notebooks/Fruitsandveg.csv", index_col=0)

# Define the path to store uploaded images
UPLOAD_FOLDER = 'upload'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

# Function to get product information based on index (row number in CSV)
def get_product_info(index):
    product_info = df.iloc[index].to_dict()
    return product_info

# Function to preprocess the image and get the index
def model_prediction(test_image):
    model = tf.keras.models.load_model("models and notebooks/trained_model.h5")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(64, 64))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # return index of max element

# Route to serve the HTML form
@app.route('/check')
def index():
    return render_template('check.html')

# Route to process the uploaded image
@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    if file:
        filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filename)
        index = model_prediction(filename)
        if 0 <= index < len(df):
            product_info = get_product_info(index)

            # Check if the product is in stock
            stock_availability = product_info['Stock Availability']
            if stock_availability > 0:
                availability = 'In Stock'
            else:
                availability = 'Sorry! Out of Stock'

            return jsonify({
                'Product Name': product_info['name'],
                'Product ID': product_info['Product ID'],
                'Category': product_info['Category'],
                'Price': product_info['Price'],
                'Unit of Measure': product_info['Unit of Measure'],
                'Description': product_info['Description'],
                'Image URL': product_info['Image URL'],
                'Stock Availability': availability,
                'Discount': product_info['Discount'],
                'Origin': product_info['Origin']
            })
        else:
            return jsonify({'error': 'Product not found'})

# Serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)



if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
