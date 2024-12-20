from model_predict import model_predict
from experiment import experiment
import argparse

def run_need_prediction():
    parser = argparse.ArgumentParser(description="Run privacy risk validation.")
    parser.add_argument("--credit_model_path", type=str, default="./model/mlp_model.pth", help="Path of the credit model.")
    parser.add_argument("--validation_data", type=str, default="./data/split/validation.csv", help="Path of the validation data to be predicted.")
    parser.add_argument("--validation_random_label_data", type=str, default="./data/predict/validation_with_random_label.csv", help="Path of the validation data with random label.")
    parser.add_argument("--validation_predictions_data", type=str, default="./data/predict/validation_with_predictions.csv", help="Path of the validation data with predictions from credit model.")
    parser.add_argument("--test_data", type=str, default="./data/split/final_test.csv", help="Path of the test data.")
    args = parser.parse_args()
    model_predict(args.credit_model_path, args.validation_data, args.validation_predictions_data)
    experiment(args.credit_model_path, args.validation_predictions_data, args.validation_random_label_data, args.test_data)

def run():
    parser = argparse.ArgumentParser(description="Run privacy risk validation.")
    parser.add_argument("--credit_model_path", type=str, default="./model/mlp_model.pth", help="Path of the credit model.")
    parser.add_argument("--validation_random_label_data", type=str, default="./data/predict/validation_with_random_label.csv", help="Path of the validation data with random label.")
    parser.add_argument("--validation_predictions_data", type=str, default="./data/predict/validation_with_predictions.csv", help="Path of the validation data with predictions from credit model.")
    parser.add_argument("--test_data", type=str, default="./data/split/final_test.csv", help="Path of the test data.")
    args = parser.parse_args()
    experiment(args.credit_model_path, args.validation_predictions_data, args.validation_random_label_data, args.test_data)

if __name__=='__main__':
    run()