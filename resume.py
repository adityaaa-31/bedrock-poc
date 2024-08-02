import boto3
import json
from PyPDF2 import PdfReader
import os

log_filename = 'chunks_log.txt'
full_text_file = 'full_text.txt'

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  
)

def read_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = " ".join(page.extract_text() for page in reader.pages)

    with open(full_text_file, 'w') as text_file:
        text_file.write(full_text)

    return full_text

def convert_to_json(chunk):

    content = f'''

    You are an AI assistant specialized in extracting information from resumes.   
    Your job is to give the output in json, do not add extra text.

    Here are some rules to follow\n

    1. Keep the details as they are given.
    2. Keep the keys in camel case
    3. Do not add links
    4. If any value is empty, do not include it in the final JSON
    5. Keep keys true to document terminology
    6. To calculate the total professional experience and average experience per company, please follow these steps:
        -Identify all unique employment periods.
        -For overlapping periods, count the time only once to avoid double-counting.
        -Calculate the total duration of non-overlapping employment.
        -Count the number of unique companies worked for.
        -Calculate the average experience per company by dividing the total non-overlapping duration by the number of companies.
        -Give the exact number of years and months in the output\n

    Carefully examine the resume and extract the following information. If any of the information is not mentioned, skip it\n
    1. Full Name
    2. Email
    3. Phone
    4. Education (Including the details)
    5. Employment / Work History (Including the details)
    6. Skills
    7. Overall experience
    8. Average exeperience in a company

    output_template = {{
    "fullName": {{ 
        "type": "string", 
        "description": "The applicant's full name" 
    }},
    "email": {{ 
        "type": "string", 
        "description": "The applicant's email address" 
    }},
    "phone": {{ 
        "type": "string", 
        "description": "The applicant's phone number" 
    }},
    "education": {{ 
        "type": "array", 
        "description": "List of the applicant's educational qualifications",
        "items": {{
        "type": "object",
        "properties": {{
            "degree": {{ 
            "type": "string", 
            "description": "The degree obtained" 
            }},
            "institution": {{ 
            "type": "string", 
            "description": "The educational institution attended" 
            }},
            "location": {{ 
            "type": "string", 
            "description": "Location of the institution" 
            }},
            "date": {{ 
            "type": "string", 
            "description": "Duration of study" 
            }},
            "details": {{ 
            "type": "array", 
            "description": "Additional details about the education",
            "items": {{ 
                "type": "string" 
            }}
            }}
        }}
        }}
    }},
    "employmentHistory": {{ 
        "type": "array", 
        "description": "List of the applicant's work experiences",
        "items": {{
        "type": "object",
        "properties": {{
            "position": {{ 
            "type": "string", 
            "description": "Job title" 
            }},
            "company": {{ 
            "type": "string", 
            "description": "Company name" 
            }},
            "location": {{ 
            "type": "string", 
            "description": "Job location" 
            }},
            "date": {{ 
            "type": "string", 
            "description": "Employment duration" 
            }},
            "responsibilities": {{ 
            "type": "array", 
            "description": "List of job responsibilities",
            "items": {{ 
                "type": "string" 
            }}
            }}
        }}
        }}
    }},
    "skills": {{ 
        "type": "array", 
        "description": "List of the applicant's skills",
        "items": {{ 
        "type": "string" 
        }}
    }},
    "overallExperience": {{ 
        "type": "string", 
        "description": "Total years of professional experience (example - 5 Years)" 
    }},
    "averageExperiencePerCompany": {{ 
        "type": "string", 
        "description": "Average years of experience per company (example - 2 Years)" 
    }}
    }}

    \nHere is the information\n
    <context> 
    {chunk}
    </context>

    '''

    messages = [
        {
            "role": "user",
            "content": content
        }
    ]
    
    body = json.dumps({
        "anthropic_version": "bedrock-2023-05-31",
        "max_tokens": 4000,
        "messages": messages,
        "temperature": 0.5,
        "top_p": 0.5,
    })

    response = bedrock.invoke_model(
        body = body,
        modelId = "anthropic.claude-3-5-sonnet-20240620-v1:0",
        contentType = "application/json",
        accept = "application/json"
    )

    response_body = json.loads(response.get('body').read())
    return response_body['content'][0]['text']


pdf_path = "/Users/adityashedge/Downloads/London-Resume-Template-Professional.pdf"

full_text_chunk = read_pdf(pdf_path)

json_result = convert_to_json(full_text_chunk)
   
output_file = os.path.splitext(pdf_path)[0] + '_output.json'
with open(output_file, 'w') as f:
    f.write(json_result)

print(f"JSON output saved to {output_file}")