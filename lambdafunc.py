import boto3
import csv

# Initialize S3 and SNS clients
s3 = boto3.client('s3')
sns = boto3.client('sns')

# Lambda handler function
def lambda_handler(event, context):
    bucket_name = 'healthdatafromsensors'  # Replace with your S3 bucket name
    object_key = 'Temp.csv'  # Replace with your CSV file key
    
    # Read the CSV file from S3
    response = s3.get_object(Bucket=bucket_name, Key=object_key)
    csv_content = response['Body'].read().decode('utf-8')
    print(csv_content)
    # Parse CSV content and check body temperature
    reader = csv.reader(csv_content.splitlines())
    next(reader)  # Skip header row
    for row in reader:
        time, body_temperature = row
        if float(body_temperature) > 99.0:
            # Send SNS notification
            sns.publish(
                TopicArn='arn:aws:sns:ap-south-1:256508801167:alertnotification',
                Subject='High Body Temperature Alert',
                Message=f'High body temperature detected at {time}: {body_temperature}°F. It is advised to kindly take rest and consult for medical care.'
            )
