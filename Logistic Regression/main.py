import preprocessing
import pretrained_model
import model_from_scratch

def main():
    """
    Main execution script to compare a professional Logistic Regression model 
    against a custom-built version from scratch.
    """
    
    # --- STEP 1: Load and Preprocess ---
    # Fetch raw data and apply cleaning (handling NaNs, encoding, and scaling)
    train, test, y_test = preprocessing.LoadData()
    train, test = preprocessing.PreprocessData(train, test, y_test)

    # Convert DataFrames to NumPy arrays for mathematical compatibility
    x_train = train.drop(columns=['Survived']).values
    y_train = train['Survived'].values
    x_test = test.drop(columns=['Survived']).values
    y_test = test['Survived'].values

    # --- STEP 2: Scikit-Learn (Benchmark) ---
    print("--- Training Scikit-Learn Model ---")
    model = pretrained_model.LoadPretrainedModel()
    model = pretrained_model.train_model(model, x_train, y_train)
    
    # Evaluate using multiple metrics (Accuracy, Precision, Recall, F1)
    accuracy, precision, recall, f1 = pretrained_model.evaluate_model(model, x_test, y_test)
    print(f"Pretrained Model - Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}, F1 Score: {f1:.4f}\n")

    # --- STEP 3: Custom Model (From Scratch) ---
    print("--- Training Custom Model (From Scratch) ---")
    # Initialize your manual Logistic Regression with specific hyperparameters
    scratch_model = model_from_scratch.LogisticRegression(
        learning_rate=0.3, 
        lambda_reg=1, 
        num_iterations=1000
    )
    
    # Run Gradient Descent
    scratch_model.fit(x_train, y_train)
    
    # Generate predictions and calculate simple accuracy
    y_pred_scratch = scratch_model.predict(x_test)
    accuracy_scratch = (y_pred_scratch == y_test).mean()
    print(f"From Scratch Model - Accuracy: {accuracy_scratch:.4f}")

if __name__ == "__main__":
    # Ensure the script only runs if executed directly (not when imported)
    main()
