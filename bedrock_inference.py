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

class PDFProcessor:
    def __init__(self):
        """Initialize with empty configuration"""
        self.input_bucket = None
        self.input_key = None
        self.uuid = None
        self.s3_client = boto3.client('s3')
        self.bedrock_client = boto3.client('bedrock-runtime')

    # ✅ Set input bucket & key
    def set_input(self, bucket: str, key: str):
        """Set the input S3 bucket and file key"""
        self.input_bucket = bucket
        self.input_key = key

    # ✅ Set UUID for tracking results
    def set_uuid(self, uuid: str):
        """Set UUID for result storage"""
        self.uuid = uuid

    def pdf_to_base64_images(self, pdf_data: bytes):
        """
        Converts each page of a PDF into a Base64-encoded image string.
        :param pdf_data: PDF file as binary data.
        :return: List of Base64-encoded image strings (one per page).
        """
        try:
            images = convert_from_bytes(pdf_data, dpi=300)  # Convert PDF pages to images
            base64_images = []

            for img in images:
                img_buffer = io.BytesIO()
                img.save(img_buffer, format="PNG")  # Save as PNG
                base64_str = base64.b64encode(img_buffer.getvalue()).decode("utf-8")
                base64_images.append(base64_str)

            return base64_images
        except Exception as e:
            logger.error(f"Error processing PDF: {str(e)}")
            raise PDFProcessingError(f"Failed to convert PDF to images: {str(e)}")

    def prepare_bedrock_payload(self, pdf_base64_images):
        """
        Prepare the payload for Bedrock inference with multiple page images.
        :param pdf_base64_images: List of Base64-encoded PDF page images.
        :return: Formatted payload for Bedrock.
        """
        try:
            if not pdf_base64_images:
                raise ValueError("No valid Base64 images found in PDF")

            messages = [{"role": "user", "content": [{"type": "text", "text": "Analyze this document and summarize the content."}]}]

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

    def write_results_to_s3(self, results):
        """
        Write processing results to S3.
        :param results: Processing results to write.
        """
        if not self.input_bucket or not self.input_key or not self.uuid:
            raise PDFProcessingError("S3 bucket, input key, or UUID is not set.")

        output_key = f"pdf_results/{self.uuid}/{os.path.splitext(os.path.basename(self.input_key))[0]}_results.json"

        try:
            self.s3_client.put_object(
                Bucket=self.input_bucket,
                Key=output_key,
                Body=json.dumps(results, indent=2),
                ContentType='application/json'
            )
            logger.info(f"Results saved to s3://{self.input_bucket}/{output_key}")
        except ClientError as e:
            logger.error(f"Failed to write results to S3: {str(e)}")
            raise PDFProcessingError(f"S3 write error: {str(e)}")

    def read_pdf_from_s3(self):
        """Download PDF file from S3."""
        if not self.input_bucket or not self.input_key:
            raise PDFProcessingError("S3 bucket or input key is not set.")

        try:
            logger.info(f"Downloading PDF from s3://{self.input_bucket}/{self.input_key}")
            response = self.s3_client.get_object(Bucket=self.input_bucket, Key=self.input_key)
            return response['Body'].read()
        except ClientError as e:
            logger.error(f"Error downloading PDF: {e}")
            raise PDFProcessingError(f"Failed to read PDF from S3: {str(e)}")

    def process(self):
        """
        Process a single PDF and analyze it using Bedrock.
        """
        if not self.input_bucket or not self.input_key or not self.uuid:
            raise PDFProcessingError("Missing required parameters: input_bucket, input_key, or uuid.")

        start_time = datetime.now().strftime("%Y%m%d_%H%M%S")

        try:
            # Read input PDF
            pdf_data = self.read_pdf_from_s3()
            if not pdf_data:
                raise PDFProcessingError("No PDF data received from S3")

            # Convert PDF to Base64 images
            pdf_base64_images = self.pdf_to_base64_images(pdf_data)

            # Prepare and call Bedrock
            logger.info(f"Calling Bedrock inference")
            bedrock_payload = self.prepare_bedrock_payload(pdf_base64_images)

            response = self.bedrock_client.invoke_model(
                modelId='anthropic.claude-3.5-sonnet-20240620-v1:0',
                body=json.dumps(bedrock_payload)
            )

            # Process results
            inference_results = json.loads(response['body'].read().decode("utf-8"))

            # Prepare final results
            results = {
                'pdf_key': self.input_key,
                'processing_start_time': start_time,
                'processing_end_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'status': 'SUCCESS',
                'analysis_results': inference_results
            }

            # Write results
            self.write_results_to_s3(results)

            return results

        except Exception as e:
            logger.error(f"Error processing PDF {self.input_key}:", exc_info=True)
            error_response = {
                'pdf_key': self.input_key,
                'processing_start_time': start_time,
                'processing_end_time': datetime.now().strftime("%Y%m%d_%H%M%S"),
                'status': 'FAILED',
                'error': str(e),
                'error_type': type(e).__name__
            }
            return error_response


if __name__ == "__main__":
    # Example Usage
    processor = PDFProcessor()

    # Set values dynamically instead of using env variables
    processor.set_input("your-input-bucket", "documents/sample.pdf")
    processor.set_uuid("123456")

    # Process PDF and analyze with Bedrock
    result = processor.process()
    logger.info(f"Processing complete: {result}")
