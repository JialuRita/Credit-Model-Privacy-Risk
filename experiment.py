import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import torch
from torch.utils.data import DataLoader, TensorDataset

from model.MLP import MLPModel

def experiment(credit_model_path, validation_predictions_data, validation_random_label_data, test_data):
    # 加载数据
    validation_with_predictions = pd.read_csv(validation_predictions_data)
    validation_with_random_label = pd.read_csv(validation_random_label_data)
    final_test = pd.read_csv(test_data)

    # 确保数据集中都包含 y_pred
    assert 'y_pred' in validation_with_predictions.columns
    assert 'y_pred' in validation_with_random_label.columns

    # 分离测试集特征和目标变量：final_test
    X_test = final_test.drop(columns=['y'])
    y_test = final_test['y']

    # 定义五种训练数据的组合比例
    combinations = {
        'a': (0.0, 1.0),  # 全部来自 validation_with_random_label
        'b': (0.25, 0.75),
        'c': (0.5, 0.5),
        'd': (0.75, 0.25),
        'e': (1.0, 0.0)   # 全部来自 validation_with_predictions
    }

    results = {}
    for combo, (pred_ratio, rand_ratio) in combinations.items():
        # 按比例采样训练数据
        data_pred = validation_with_predictions.sample(frac=pred_ratio, random_state=42)
        data_rand = validation_with_random_label.sample(frac=rand_ratio, random_state=42)
        
        # 合并数据
        combined_data = pd.concat([data_pred, data_rand], axis=0).reset_index(drop=True)
        
        # 分离特征和目标变量
        X_train = combined_data.drop(columns=['y', 'y_pred'])
        y_train = combined_data['y_pred']

        # 训练二分类器
        model = RandomForestClassifier(random_state=42, n_estimators=300)
        model.fit(X_train, y_train)

        # 在测试集上预测
        y_pred_test = model.predict(X_test)

        # 计算准确率和报告
        accuracy = accuracy_score(y_test, y_pred_test)
        report = classification_report(y_test, y_pred_test)

        print(f"Combination {combo}: Accuracy = {accuracy:.4f}")

        # 计算征信模型输出的授信率
        print(f"使用data combination {combo}训练的分类器筛选后的授信率：", end="")
        credit_rate = classify_filtered_credit_rate(credit_model_path, model, final_test)
        
        # 保存结果
        results[combo] = {
            'accuracy': accuracy,
            'report': report,
            'credit_rate': credit_rate
        }

    print("不使用分类器筛选的授信率：", end="")
    base_credit_rate(credit_model_path, final_test)
    
    # 根据分类器的准确率以及征信模型输出的授信率提升情况评估风险    
    val_risk_by_accuracy_and_credit_rate(results)

def load_credit_model(credit_model_path, final_test):
    # 分离测试集特征和目标变量
    X_test = torch.tensor(final_test.drop(columns=['y']).values, dtype=torch.float32)
    y_test = torch.tensor(final_test['y'].values, dtype=torch.long)
    # 加载credit model
    input_size = X_test.shape[1]
    credit_model = MLPModel(input_size)
    # 加载训练好的模型参数
    credit_model.load_state_dict(torch.load(credit_model_path))
    credit_model.eval()
    return credit_model

def base_credit_rate(credit_model_path, final_test):
    credit_model = load_credit_model(credit_model_path, final_test)
    # 分离测试集特征和目标变量
    X_test = torch.tensor(final_test.drop(columns=['y']).values, dtype=torch.float32)
    y_test = torch.tensor(final_test['y'].values, dtype=torch.long)
    # baseline: without classify filtered
    cal_credit_rate(credit_model, X_test, y_test)

def classify_filtered_credit_rate(credit_model_path, model, final_test):
    credit_model = load_credit_model(credit_model_path, final_test)
    # 使用分类模型进行预测并筛选数据
    y_pred_test = model.predict(final_test.drop(columns=['y']))
    # 筛选出预测为 1 的数据
    selected_indices = [i for i, pred in enumerate(y_pred_test) if pred == 1]
    new_test_data = final_test.iloc[selected_indices]
    # print(f"Filtered Test Data: {len(new_test_data)} samples selected from {len(final_test)}")
    # 分离测试集特征和目标变量
    X_test = torch.tensor(new_test_data.drop(columns=['y']).values, dtype=torch.float32)
    y_test = torch.tensor(new_test_data['y'].values, dtype=torch.long)
    # experiments: with classify filtered
    credit_rate = cal_credit_rate(credit_model, X_test, y_test)
    return credit_rate

def cal_credit_rate(credit_model, X_test, y_test):
    batch_size = 64
    final_test_dataset = TensorDataset(X_test, y_test)
    final_test_loader = DataLoader(final_test_dataset, batch_size=batch_size, shuffle=False)
    y_pred = []
    # 通过模型预测
    credit_model.eval()
    with torch.no_grad():
        for X_batch, _ in final_test_loader:
            outputs = credit_model(X_batch)
            _, predicted = torch.max(outputs, 1)
            y_pred.extend(predicted.tolist())
    # 计算 y_pred 中值为 1 的比例
    credit_rate = y_pred.count(1) / len(y_pred)
    print(f"Credit Rate (proportion of 1 in predictions): {credit_rate:.4f}")
    return credit_rate

def val_risk_by_accuracy_and_credit_rate(results):
    # 输出所有组合的结果
    baseline_accuracy = results['a']['accuracy']
    baseline_credit_rate = results['a']['credit_rate']
    # 评估风险
    print("\nRisk Assessment:")
    for combo, result in results.items():
        if combo == 'a':
            continue
        accuracy_improvement = result['accuracy'] - baseline_accuracy
        credit_rate_change = result['credit_rate'] - baseline_credit_rate
        print(f"Combination {combo}: Accuracy Improvement: {accuracy_improvement:.4f}, Credit Rate Change: {credit_rate_change:.4f}")

    if any(result['accuracy'] - baseline_accuracy > 0.2 for combo, result in results.items() if combo != 'a'):
        print("\nRisk Detected: Credit model outputs significantly improve the attacker's classifier accuracy.")
        risk_detected = True
    
    if any(result['credit_rate'] - baseline_credit_rate > 0.2 for combo, result in results.items() if combo != 'a'):
        print("\nRisk Detected: Credit model outputs significantly increase the credit rate.")
        risk_detected = True

    if not risk_detected:
        print("\nNo Significant Risk Detected: Credit model outputs do not significantly affect accuracy or credit rate.")
