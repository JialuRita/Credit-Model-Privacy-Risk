import pandas as pd
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# 加载数据集
data = pd.read_csv('./data/bank_data.csv')

# 查看数据基本信息
print(data.info())

# 将目标变量 'y' 转换为数值格式
data['y'] = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

# 列出需要进行 OneHot 编码的类别变量
categorical_columns = ['job', 'marital', 'education', 'default', 'housing', 'loan', 'contact', 'month', 'poutcome']

# 使用 pandas 的 get_dummies 进行 OneHot 编码，并将虚拟变量值设置为 0 和 1
data = pd.get_dummies(data, columns=categorical_columns, drop_first=True)
data = data.astype({col: 'int32' for col in data.columns if col.startswith(tuple(categorical_columns))})

# 检查数值变量并进行标准化
numerical_columns = ['age', 'balance', 'day', 'duration', 'campaign', 'pdays', 'previous']
scaler = StandardScaler()
data[numerical_columns] = scaler.fit_transform(data[numerical_columns])

# 检查缺失值
if data.isnull().sum().any():
    print("缺失值列:")
    print(data.isnull().sum()[data.isnull().sum() > 0])
    # 填充缺失值（此处用中位数填充数值变量，类别变量用众数填充）
    for col in data.columns:
        if data[col].dtype in ['float64', 'int64']:
            data[col].fillna(data[col].median(), inplace=True)
        else:
            data[col].fillna(data[col].mode()[0], inplace=True)

# 按照目标数量划分数据集
train, temp = train_test_split(data, train_size=20000, stratify=data['y'], random_state=42)
validation, temp = train_test_split(temp, train_size=10000, stratify=temp['y'], random_state=42)
random_label, final_test = train_test_split(temp, train_size=10000, stratify=temp['y'], random_state=42)

# 检查每个数据集中 y 的分布
print("Train y 分布:")
print(train['y'].value_counts(normalize=True))
print("Validation y 分布:")
print(validation['y'].value_counts(normalize=True))
print("Final Test y 分布:")
print(final_test['y'].value_counts(normalize=True))
print("Random Label y 分布:")
print(random_label['y'].value_counts(normalize=True))

# 保存数据集
train.to_csv('./data/split/train.csv', index=False)
validation.to_csv('./data/split/validation.csv', index=False)
final_test.to_csv('./data/split/final_test.csv', index=False)
random_label.to_csv('./data/split/random_label.csv', index=False)

print("数据集划分完成，已保存为 train.csv, validation.csv, final_test.csv, random_label.csv")
