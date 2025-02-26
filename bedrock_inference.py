import boto3
import json
import os
import logging
import datetime
import base64
import io
from pdf2image import convert_from_bytes
from PIL import Image
from botocore.exceptions import ClientError
from typing import Dict, Any, Union
from datetime import datetime

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class PDFProcessingError(Exception):
    """Custom exception for PDF processing errors"""
    pass

# Initialize AWS clients
def init_aws_clients():
    """Initialize AWS clients with error handling"""
    try:
        return {
            's3_client': boto3.client('s3'),
            'bedrock': boto3.client('bedrock-runtime')
        }
    except Exception as e:
        logger.error(f"Failed to initialize AWS clients: {str(e)}")
        raise

# Configuration
def get_config():
    """Get configuration from environment variables"""
    return {
        'input_bucket': os.environ.get('INPUT_BUCKET'),
        'uuid': os.environ.get('UUID'),
        'input_key': os.environ.get('INPUT_KEY'),
        'local_input': '/tmp/pdf_file.pdf'
    }

def pdf_to_base64_images(pdf_data: bytes):
    """
    Converts each page of a PDF into a Base64-encoded image string.
    
    :param pdf_data: PDF file as binary data.
    :return: List of Base64-encoded image strings (one per page).
    """
    try:
        images = convert_from_bytes(pdf_data, dpi=300)  # Render PDF pages as images
        base64_images = []

        for img in images:
            # Convert image to Base64
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="PNG")  # Save as PNG
            base64_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")

            base64_images.append(base64_str)

        return base64_images
    except Exception as e:
        logger.error(f"Error processing PDF: {str(e)}")
        raise PDFProcessingError(f"Failed to convert PDF to images: {str(e)}")

def prepare_bedrock_payload(pdf_base64_images):
    """
    Prepare the payload for Bedrock inference with multiple page images.
    
    :param pdf_base64_images: List of Base64-encoded PDF page images.
    :return: Formatted payload for Bedrock.
    """
    try:
        if not pdf_base64_images:
            raise ValueError("No valid Base64 images found in PDF")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Analyze this document and summarize the content."}
                ]
            }
        ]

        # Attach all pages as Base64 images
        for img_str in pdf_base64_images:
            messages[0]["content"].append({
                "type": "image",
                "source": {
                    "type": "base64",
                    "media_type": "image/png",
                    "data": img_str
                }
            })

        return {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": 2048,
            "messages": messages
        }
    except (AttributeError, ValueError) as e:
        logger.error(f"Error preparing Bedrock payload: {str(e)}")
        raise PDFProcessingError(f"Failed to prepare PDF payload: {str(e)}")

def write_results_to_s3(s3_client, results, bucket, key):
    """
    Write processing results to S3.
    
    :param s3_client: Boto3 S3 client.
    :param results: Processing results to write.
    :param bucket: Destination bucket.
    :param key: Destination key.
    """
    try:
        s3_client.put_object(
            Bucket=bucket,
            Key=key,
            Body=json.dumps(results, indent=2),
            ContentType='application/json'
        )
    except ClientError as e:
        logger.error(f"Failed to write results to S3: {str(e)}")
        raise PDFProcessingError(f"S3 write error: {str(e)}")

def read_pdf_from_s3(s3_client, config):
    """Download PDF file from S3."""
    try:
        logger.info(f"Downloading PDF from s3://{config['input_bucket']}/{config['input_key']}")
        response = s3_client.get_object(
            Bucket=config['input_bucket'],
            Key=config['input_key']
        )
        return response['Body'].read()
    except ClientError as e:
        logger.error(f"Error downloading PDF: {e}")
        raise PDFProcessingError(f"Failed to read PDF from S3: {str(e)}")

def process_pdf():
    """
    Process a single PDF and analyze it using Bedrock.
    """
    start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

    try:
        # Initialize AWS clients
        aws_clients = init_aws_clients()
        config = get_config()

        # Read input PDF
        pdf_data = read_pdf_from_s3(aws_clients['s3_client'], config)
        if not pdf_data:
            raise PDFProcessingError("No PDF data received from S3")

        # Convert PDF to Base64 images
        pdf_base64_images = pdf_to_base64_images(pdf_data)

        # Prepare and call Bedrock
        logger.info(f"Calling Bedrock inference")
        bedrock_payload = prepare_bedrock_payload(pdf_base64_images)

        response = aws_clients['bedrock'].invoke_model(
            modelId='anthropic.claude-3.5-sonnet-20240620-v1:0',
            body=json.dumps(bedrock_payload)
        )

        # Process results
        inference_results = json.loads(response['body'].read().decode("utf-8"))

        # Prepare final results
        results = {
            'pdf_key': config['input_key'],
            'processing_start_time': start_time,
            'processing_end_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'status': 'SUCCESS',
            'analysis_results': inference_results
        }

        # Write results
        output_key = os.path.join(
            "pdf_results",
            config['uuid'],
            f"{os.path.splitext(os.path.basename(config['input_key']))[0]}_results.json"
        )
        
        logger.info(f"Writing results to {output_key} in bucket {config['input_bucket']}")
        write_results_to_s3(aws_clients['s3_client'], results, config['input_bucket'], output_key)

        return results

    except Exception as e:
        logger.error(f"Error processing PDF {config['input_key']}:", exc_info=True)
        error_response = {
            'pdf_key': config['input_key'],
            'processing_start_time': start_time,
            'processing_end_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
            'status': 'FAILED',
            'error': str(e),
            'error_type': type(e).__name__
        }
        return error_response

if __name__ == "__main__":
    result = process_pdf()
    logger.info(f"Processing complete: {result}")
