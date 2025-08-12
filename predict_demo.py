from src.pipeline.prediction_pipeline import VehicleData, VehicleDataClassifier

# Create a VehicleData object with some dummy values
vehicle_data = VehicleData(
    Gender="Male",
    Age=35,
    Region_Code="2",
    Previously_Insured=0,
    Annual_Premium=3500.5,
    Policy_Sales_Channel="26",
    Vintage=200,
    Vehicle_Age=1,
    Vehicle_Damage="Yes"
)

# Convert the input to a DataFrame
vehicle_df = vehicle_data.get_vehicle_input_data_frame()

# Initialize the classifier
classifier = VehicleDataClassifier()

# Run prediction
predictions, latency, rate = classifier.predict(vehicle_df)

# Print results
print("Predictions:", predictions)
print(f"Latency: {latency:.4f} sec")
print(f"Throughput: {rate:.2f} predictions/sec")

output_path, latency, rate = classifier.batch_predict('batch_test.csv')

# Print results
print(f"Batch Latency: {latency:.4f} sec")
print(f"Batch Throughput: {rate:.2f} predictions/sec")
