# Analyzing-Customer-Engagemnet


### Detailed Analysis of the Project: "Analyzing Customer Engagement - Total Transactions and Behavior Insights"

#### 1. **Project Objective**
The primary goal of this project is to analyze customer engagement by focusing on their transaction behavior. This involves assessing total transactions, identifying key behavioral insights, and eventually building a machine learning model to forecast future trends. The tools used for this project include PySpark for data processing, Matplotlib for visualization, and Spark MLlib for machine learning.

#### 2. **Data Collection**
The project begins with collecting customer and transaction data, typically stored in CSV files. This data includes essential details such as customer IDs, names, transaction dates, transaction amounts, and other relevant metrics.

- **Loading Data**: The data is loaded into Spark DataFrames using PySpark, which allows efficient handling of large datasets.

```python
from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder \
    .appName("Customer_Engagement_Analysis") \
    .config("spark.executor.memory", "4g") \
    .config("spark.executor.cores", "4") \
    .getOrCreate()

# Load data into Spark DataFrames
nasabah_spark_df = spark.read.csv('nasabah.csv', header=True, inferSchema=True)
transaksi_spark_df = spark.read.csv('transaksi.csv', header=True, inferSchema=True)
```

#### 3. **Data Exploration**
After loading the data, the next step is to explore the dataset. This includes understanding the structure, identifying missing values, and analyzing the data distribution.

- **Schema Check**: Print the schema of the DataFrames to verify the data types.
- **Data Preview**: Display the first few rows to ensure data has been loaded correctly.

```python
# Display schemas
nasabah_spark_df.printSchema()
transaksi_spark_df.printSchema()

# Show a few rows of the data
nasabah_spark_df.show()
transaksi_spark_df.show()
```

#### 4. **Data Processing and SQL Queries**
To derive insights, various SQL queries are executed on the data. One of the key queries involves joining the customer and transaction data to calculate the total transactions per customer.

```python
simple_query = """
SELECT
    t.id_nasabah AS CustomerID,
    n.nama AS CustomerName,
    COUNT(t.id_transaksi) AS Total_Transactions
FROM
    transaksi t
JOIN
    nasabah n ON t.id_nasabah = n.id_nasabah
GROUP BY
    t.id_nasabah, n.nama
"""
# Execute the query and convert the result to a Pandas DataFrame
simple_result_df = spark.sql(simple_query)
pandas_df = simple_result_df.toPandas()
```

This query helps identify customers with the highest engagement, as measured by the number of transactions.

#### 5. **Visualization**
Using Matplotlib, the data is visualized to provide a clear picture of customer behavior. Visualizations such as bar charts are created to display the total transactions per customer.

```python
import matplotlib.pyplot as plt

# Plotting the total transactions per customer
plt.figure(figsize=(10, 6))
plt.barh(pandas_df['CustomerName'], pandas_df['Total_Transactions'], color='skyblue')
plt.xlabel('Total Transactions')
plt.ylabel('Customer Name')
plt.title('Total Transactions per Customer')
plt.show()
```

#### 6. **Machine Learning Model**
After visualizing the data, a machine learning model is built to forecast future transactions or spending behavior. A linear regression model is chosen for this purpose.

```python
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.regression import LinearRegression

# Prepare features and labels
assembler = VectorAssembler(inputCols=["timestamp_unix"], outputCol="features")
feature_df = assembler.transform(data_df)

# Train-test split
train_df, test_df = feature_df.randomSplit([0.8, 0.2])

# Initialize and train the Linear Regression model
lr = LinearRegression(featuresCol="features", labelCol="label")
lr_model = lr.fit(train_df)
```

#### 7. **Model Evaluation**
The model is then evaluated on the test data to check its accuracy and effectiveness. Key metrics like Root Mean Square Error (RMSE) and R-squared are calculated.

```python
# Predictions
predictions = lr_model.transform(test_df)

# Evaluate the model
evaluator = RegressionEvaluator(predictionCol="prediction", labelCol="label", metricName="rmse")
rmse = evaluator.evaluate(predictions)
print(f"Root Mean Square Error (RMSE): {rmse}")
```

#### 8. **Conclusion and Insights**
- **Data Insights**: The initial data exploration and SQL queries reveal important insights about customer engagement. For instance, the customers with the highest number of transactions are identified, helping the business focus on retaining these high-value customers.
  
- **Visual Analysis**: The visualizations provide an easy-to-understand summary of customer behavior, which is crucial for stakeholders who may not be familiar with the technical details.
  
- **Predictive Modeling**: The machine learning model offers a way to predict future customer behavior, enabling proactive engagement strategies. The model's performance, as indicated by metrics like RMSE, shows its reliability in making accurate predictions.

- **Business Application**: The insights gained from this project can be used to tailor marketing campaigns, improve customer retention, and optimize the overall customer experience.

#### 9. **Final Thoughts**
This project effectively combines data processing, visualization, and machine learning to provide a comprehensive analysis of customer engagement. The approach is scalable and can be applied to various domains where understanding customer behavior is critical.
