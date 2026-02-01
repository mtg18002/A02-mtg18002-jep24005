# A02-mtg18002-jep24005
## Ping pong assignment
Jiben Gigi and Matt Gostkowski

For this assignment, we will practice GitHub collaboration to train a neural network on the California housing dataset. This neural network regression model will use the numerical predictors available in the CA housing dataset to predict MedHouseVal. We will also use train and test predictions to evaluate model fit and tune hyperparameters.  

### To run the code:
First, make sure you are in the correct working directory. 

Then, install the packages in the requirements.txt file:
pip install -r requirements.txt

Run the code:
python .\src\CA_housing_nn.py

### Workflow
* Data Loading: Import the California Housing dataset and separate features and target (MedHouseVal).
* Data Splitting: Split the data into training (80%), validation (10%), and test (10%) sets.
* Preprocessing: Standardize features using StandardScaler to ensure effective neural network training.
* Model Training: Train an MLPRegressor with ReLU activation and early stopping to prevent overfitting.
* Evaluation: Measure performance using R², MAE, and MAPE on training and test data.
* Visualization: Plot predicted vs. actual values and the training loss curve to assess model fit and convergence.
