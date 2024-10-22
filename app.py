from flask import Flask, render_template, request
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier


X_train = np.array([[1, 1, 1, 1, 1, 1, 1, 1], 
                    [2, 2, 2, 2, 2, 2, 2, 2], 
                    [3, 3, 3, 3, 3, 3, 3, 3], 
                    [4, 4, 4, 4, 4, 4, 4, 4]])
y_train = np.array([0, 1, 1, 0])  


knn = KNeighborsClassifier(n_neighbors=3)
nb = GaussianNB()
dt = DecisionTreeClassifier()


knn.fit(X_train, y_train)
nb.fit(X_train, y_train)
dt.fit(X_train, y_train)


app = Flask(__name__)


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
  
    clump_thickness = int(request.form['clump_thickness'])
    uniformity_cell_size = int(request.form['uniformity_cell_size'])
    uniformity_cell_shape = int(request.form['uniformity_cell_shape'])
    marginal_adhesion = int(request.form['marginal_adhesion'])
    single_epithelial_cell_size = int(request.form['single_epithelial_cell_size'])
    bland_chromatin = int(request.form['bland_chromatin'])
    normal_nucleoli = int(request.form['normal_nucleoli'])
    mitoses = int(request.form['mitoses'])

   
    model_choice = request.form['classifier']

 
    features = np.array([[clump_thickness, 
                          uniformity_cell_size, 
                          uniformity_cell_shape, 
                          marginal_adhesion,
                          single_epithelial_cell_size,
                          bland_chromatin,
                          normal_nucleoli,
                          mitoses]])

    
    if model_choice == 'knn':
        prediction = knn.predict(features)[0]
        classifier = "K-Nearest Neighbors"
    elif model_choice == 'nb':
        prediction = nb.predict(features)[0]
        classifier = "Naive Bayes"
    elif model_choice == 'dt':
        prediction = dt.predict(features)[0]
        classifier = "Decision Tree"
    else:
        prediction = 'Invalid model choice'

  
    result = 'Malignant' if prediction == 1 else 'Benign'
    return render_template('result.html', 
                                prediction=result, 
                                classifier=classifier,
                                clump_thickness=clump_thickness,
                                uniformity_cell_size=uniformity_cell_size,
                                uniformity_cell_shape=uniformity_cell_shape,
                                marginal_adhesion=marginal_adhesion,
                                single_epithelial_cell_size=single_epithelial_cell_size,
                                bland_chromatin=bland_chromatin,
                                normal_nucleoli=normal_nucleoli,
                                mitoses=mitoses)


if __name__ == '__main__':
    app.run(debug=True)
    