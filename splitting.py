import boto3
import json
from PyPDF2 import PdfReader
import os
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

# logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
# logger = logging.getLogger(__name__)
log_filename = 'chunks_log.txt'
full_text_file = 'full_text.txt'

bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'  
)

def read_and_split_pdf(file_path):
    reader = PdfReader(file_path)
    full_text = " ".join(page.extract_text() for page in reader.pages)

    with open(full_text_file, 'w') as text_file:
        text_file.write(full_text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=4000,
        chunk_overlap=0,
        length_function=len
    )

    chunks = text_splitter.split_text(full_text)
        
    # Log each chunk
    with open(log_filename, 'w') as log_file:
        for i, chunk in enumerate(chunks):
            # Write the chunk information to the file
            log_file.write(f"Chunk {i+1}:\n{chunk}\n\n")
    
    return chunks

def convert_to_json(chunk):

    content = f'''

    You are an AI assistant specialized in extracting information from visa applications.   
    Your job is to give the output in json, do not add extra text.
    
    Here are some rules to follow\n

    1. Keep the keys in camel case 
    2. Do not change the key and value names of the details, keep them as they are given.
    3. The date must be ISO format 
    4. The country codes should be two letters (eg. INDIA will be IN)
    5. Do not group addresses
    6. Please include all the questions and their answers
    7. Do not change the format of the answers, keep the answers as they are given.
    8. Do not include keys if their values are null.
    9. You will be given a template that you can follow for the json output.
    10. If you find the key map it's value in the json.
 
    Sample Output\n

    output_template = {{ 
      "matter.additionalAttributes.havereadandUnderstoodInformation": {{ "type": "string", "description": "Whether the applicant has read and understood the information" }},
      "matter.additionalAttributes.haveprovidedcompleteAndCorrectInformation": {{ "type": "string", "description": "Whether the applicant has provided complete and correct information" }},
      "matter.additionalAttributes.willinformtheDepartment": {{ "type": "string", "description": "Whether the applicant will inform the department of any changes" }},
      "matter.additionalAttributes.havereadinformationContainedPrivacyNotice": {{ "type": "string", "description": "Whether the applicant has read the privacy notice" }},
      "matter.additionalAttributes.understandthatTheDepartment": {{ "type": "string", "description": "Whether the applicant understands that the department may use the information" }},
      "matter.additionalAttributes.organisationconsentsToSponsor": {{ "type": "string", "description": "Whether the organization consents to sponsor" }},
      "matter.additionalAttributes.willComply": {{ "type": "string", "description": "Whether the applicant will comply with the requirements" }},
      "matter.additionalAttributes.hasTheApplicantRecoveredPerson": {{ "type": "string", "description": "Whether the applicant has recovered a person" }},
      "matter.additionalAttributes.intendToSponsorWorkers": {{ "type": "string", "description": "Number of workers the applicant intends to sponsor" }},
      "matter.additionalAttributes.haveVisaGrant": {{ "type": "string", "description": "Whether the applicant has a visa grant" }},
      "matter.additionalAttributes.hasTheApplicantTakenActionToAnotherPerson": {{ "type": "string", "description": "Whether the applicant has taken action against another person" }},
      "matter.additionalAttributes.selectAccreditationCategorys": {{ "type": "string", "description": "Selected accreditation categories" }},
      "matter.additionalAttributes.selectTypePartner": {{ "type": "string", "description": "Selected type of partner" }},
      "matter.additionalAttributes.hasTheApplicantBeenSbsInPast": {{ "type": "string", "description": "Whether the applicant has been an SBS in the past" }},
      "matter.additionalAttributes.giveDetailsAdverseInfo": {{ "type": "string", "description": "Details of any adverse information" }},
      "matter.additionalAttributes.understandFigerPrintFacialImageGivenToAustralianGovernment": {{ "type": "string", "description": "Understanding of fingerprint and facial image usage by the Australian government" }},
      "matter.additionalAttributes.has18oOverReadAndAgreed": {{ "type": "string", "description": "Whether the applicant is 18 or over and has read and agreed" }},
      "matter.additionalAttributes.hasTheApplicantTransfer": {{ "type": "string", "description": "Whether the applicant has transferred any assets" }},
      "matter.additionalAttributes.personnotincludedRefusedApplication": {{ "type": "string", "description": "Whether a person not included has been refused application" }},
      "matter.additionalAttributes.hasTheApplicantTakenAnyActionSought": {{ "type": "string", "description": "Whether the applicant has taken any action sought" }},
      "matter.additionalAttributes.continue": {{ "type": "string", "description": "Continuation status" }},
      "matter.additionalAttributes.understandUseOfPersonalInformation": {{ "type": "string", "description": "Understanding of personal information usage" }},
      "matter.additionalAttributes.hasReadPrivacyNotice": {{ "type": "string", "description": "Whether the applicant has read the privacy notice" }},
      "matter.additionalAttributes.startedOperateAustralia": {{ "type": "string", "description": "Whether the applicant has started to operate in Australia" }},
      "matter.additionalAttributes.hasTheapplicantPreviouslyHadLabourAgreement": {{ "type": "string", "description": "Whether the applicant has previously had a labour agreement" }},
      "matter.additionalAttributes.giveDetailedExplanation": {{ "type": "string", "description": "Detailed explanation of the applicant's intentions" }},
      "matter.additionalAttributes.giveConsentToFigerPrintFacialImageUnder16": {{ "type": "string", "description": "Consent for fingerprint and facial image for under 16" }},
      "matter.additionalAttributes.understandVisaCeasesEffectiveBeUnlawful": {{ "type": "string", "description": "Understanding that visa ceases to be effective if unlawful" }},
      "matter.additionalAttributes.haveAnyApplicantCircumstances": {{ "type": "string", "description": "Whether there are any applicant circumstances" }},
      "matter.additionalAttributes.misleadingInformationRefusedApplication": {{ "type": "string", "description": "Whether misleading information has led to refused application" }},
      "matter.additionalAttributes.isThereAnyAdverseInformation": {{ "type": "string", "description": "Whether there is any adverse information" }},
      "matter.additionalAttributes.hasTheApplicantRecovered": {{ "type": "string", "description": "Whether the applicant has recovered any assets" }},
      "matter.additionalAttributes.giveConsentToDiscloseBiometricInformationForMigration": {{ "type": "string", "description": "Consent to disclose biometric information for migration" }},
      "matter.additionalAttributes.hasProvideCompleteInformation": {{ "type": "string", "description": "Whether complete information has been provided" }},
      "matter.additionalAttributes.doesTheApplicantReapply": {{ "type": "string", "description": "Whether the applicant is reapplying" }},
      "matter.additionalAttributes.haveComplied": {{ "type": "string", "description": "Whether the applicant has complied with requirements" }},
      "matter.additionalAttributes.informDepartmentOfChangeOfCircumstance": {{ "type": "string", "description": "Agreement to inform department of change in circumstances" }},
      "matter.additionalAttributes.isTheApplicantSubstantiveVisa": {{ "type": "string", "description": "Whether the applicant is on a substantive visa" }},
      "matter.additionalAttributes.hasTheApplicantTakenAnyAction": {{ "type": "string", "description": "Whether the applicant has taken any action" }},
      "matter.additionalAttributes.hasUnderstoodInformationProvided": {{ "type": "string", "description": "Whether the applicant has understood the information provided" }},
      "matter.additionalAttributes.hadMadeArrangementForHealthInsurance": {{ "type": "string", "description": "Whether arrangements for health insurance have been made" }},
      "matter.additionalAttributes.intendSponsorWorkerBusiness": {{ "type": "string", "description": "Intention to sponsor worker business" }},
      "matter.additionalAttributes.istheApplicantCurrently": {{ "type": "string", "description": "Current status of the applicant" }},
      "matter.additionalAttributes.doesTheApplicantContinue": {{ "type": "string", "description": "Whether the applicant continues" }},
      "matter.additionalAttributes.applicationID": {{ "type": "string", "description": "Application ID" }},
      "matter.additionalAttributes.giveConsentToFigerPrintFacialImage": {{ "type": "string", "description": "Consent for fingerprint and facial image" }},
      "matter.additionalAttributes.haveAllAustralians": {{ "type": "string", "description": "Whether all Australians have been considered" }},
      "matter.additionalAttributes.misleadingInformationAfterVisaGrantRefusedApplication": {{ "type": "string", "description": "Whether misleading information after visa grant has led to refused application" }},
      "matter.additionalAttributes.giveConsentToDiscloseBiometricInformationForVisaGrant": {{ "type": "string", "description": "Consent to disclose biometric information for visa grant" }},
      "matter.additionalAttributes.theApplicantUnderstand": {{ "type": "string", "description": "Whether the applicant understands" }},
      "matter.additionalAttributes.intendToTravelLaissezPasser": {{ "type": "string", "description": "Intention to travel with Laissez-Passer" }},
      "matter.additionalAttributes.dobPrimaryApplicant": {{ "type": "string", "description": "Date of birth of primary applicant" }},
      "matter.additionalAttributes.giveConsentToACIC": {{ "type": "string", "description": "Consent given to ACIC" }},
      "matter.additionalAttributes.mustAbideToVisaConditions": {{ "type": "string", "description": "Agreement to abide by visa conditions" }},
      "matter.additionalAttributes.referenceNumberType1": {{ "type": "string", "description": "Reference number type 1" }},
      "matter.additionalAttributes.hastheapplicantDepartment": {{ "type": "string", "description": "Whether the applicant has a department" }},
      "matter.additionalAttributes.hasTheApplicantTaken": {{ "type": "string", "description": "Whether the applicant has taken action" }},
      "matter.additionalAttributes.authorises": {{ "type": "string", "description": "Authorizations given" }},
      "matter.additionalAttributes.hasTheApplicantPractioner": {{ "type": "string", "description": "Whether the applicant has a practitioner" }},
      "matter.additionalAttributes.engageAllTemporarys": {{ "type": "string", "description": "Engagement of all temporaries" }},
      "matter.additionalAttributes.declaresNonEngagedInContravention": {{ "type": "string", "description": "Declaration of non-engagement in contravention" }},
      "matter.additionalAttributes.authorizedPersonEmailAddress": {{ "type": "string", "description": "Email address of authorized person" }},
      "matter.additionalAttributes.hasTheApplicantRecruiting": {{ "type": "string", "description": "Whether the applicant is recruiting" }},
      "matter.SponsorCompanyName": {{ "type": "string", "description": "Name of the sponsor company" }},
      "matter.CompanyName": {{ "type": "string", "description": "Name of the company" }},
      "matter.ApplicantGivenNames": {{ "type": "string", "description": "Given names of the applicant" }},
      "matter.ApplicantSurname": {{ "type": "string", "description": "Surname of the applicant" }},
      "individual.additionalAttributes.applicantLanguageSkillsDetails": {{ "type": "array", "description": "Details of applicant's language skills" }},
      "individual.additionalAttributes.dateOfIntendedMarriage": {{ "type": "string", "description": "Date of intended marriage" }},
      "individual.additionalAttributes.doesThisApplicantHavePreviousPassports": {{ "type": "string", "description": "Whether the applicant has previous passports" }},
      "individual.additionalAttributes.otherNamesDetails": {{ "type": "array", "items": {{
        "type": "object",
        "properties": {{
          "otherNamesDetailsReasonForNameChange": {{ "type": "string", "description": "Reason for name change" }},
          "otherNamesDetailsFirstName": {{ "type": "string", "description": "First name in other names" }},
          "otherNamesDetailsLastName": {{ "type": "string", "description": "Last name in other names" }}
        }}
      }} }},
      "individual.additionalAttributes.relationshipStatus": {{ "type": "string", "description": "Relationship status" }},
      "individual.additionalAttributes.citizenOfSelectedCountry": {{ "type": "string", "description": "Whether citizen of selected country" }},
      "individual.additionalAttributes.passportExpiryDate": {{ "type": "string", "description": "Passport expiry date" }},
      "individual.additionalAttributes.citizenCountry": {{ "type": "array", "items": {{
        "type": "object",
        "properties": {{
          "countriesOfCitizen": {{ "type": "string", "description": "Countries of citizenship" }}
        }}
      }} }},
      "individual.additionalAttributes.cityOfBirth": {{ "type": "string", "description": "City of birth" }},
      "individual.additionalAttributes.otherIdentityDocumentDetails": {{ "type": "array", "items": {{
        "type": "object",
        "properties": {{
          "otherIdentityDocumentDetailsCountryOfIssue": {{ "type": "string", "description": "Country of issue for other identity document" }},
          "otherIdentityDocumentDetailsLastName": {{ "type": "string", "description": "Last name on other identity document" }},
          "otherIdentityDocumentDetailsIdentificationNumber": {{ "type": "string", "description": "Identification number on other identity document" }},
          "otherIdentityDocumentDetailsFirstName": {{ "type": "string", "description": "First name on other identity document" }},
          "otherIdentityDocumentDetailsTypeOfDocument": {{ "type": "string", "description": "Type of other identity document" }}
        }}
      }} }},
      "individual.additionalAttributes.relationshipStatusRelation": {{ "type": "string", "description": "Relationship status relation" }},
      "individual.additionalAttributes.countryOfBirth": {{ "type": "string", "description": "Country of birth" }},
      "individual.additionalAttributes.passportIssueDate": {{ "type": "string", "description": "Passport issue date" }},
      "individual.additionalAttributes.otherTravelDocumentDetails": {{ "type": "array", "items": {{
        "type": "object",
        "properties": {{
          "otherTravelDocumentLastName": {{ "type": "string", "description": "Last name on other travel document" }},
          "otherTravelDocumentNumber": {{ "type": "string", "description": "Number of other travel document" }},
          "otherTravelDocumentIssuingCountry": {{ "type": "string", "description": "Issuing country of other travel document" }},
          "otherTravelDocumentIssuingAuthority": {{ "type": "string", "description": "Issuing authority of other travel document" }},
          "otherTravelDocumentNationality": {{ "type": "string", "description": "Nationality on other travel document" }},
          "otherTravelDocumentSex": {{ "type": "string", "description": "Sex on other travel document" }},
          "otherTravelDocumentExpiryDate": {{ "type": "string", "description": "Expiry date of other travel document" }},
          "otherTravelDocumentFirstName": {{ "type": "string", "description": "First name on other travel document" }},
          "otherTravelDocumentIssueDate": {{ "type": "string", "description": "Issue date of other travel document" }},
          "otherTravelDocumentDateOfBirth": {{ "type": "string", "description": "Date of birth on other travel document" }}
        }}
      }} }},
      "individual.additionalAttributes.stateOfBirth": {{ "type": "string", "description": "State of birth" }},
      "individual.additionalAttributes.otherTravelDocument": {{ "type": "string", "description": "Other travel document" }},
      "individual.additionalAttributes.healthExaminationForAustralianVisa": {{ "type": "string", "description": "Health examination for Australian visa" }},
      "individual.additionalAttributes.sex": {{ "type": "string", "description": "Sex of the individual" }},
      "individual.additionalAttributes.nationalIdCardDetails": {{ "type": "array", "items": {{
      "type": "object",
      "properties": {{
        "nationalIdCardIdentityNumber": {{ "type": "string", "description": "Identity number on national ID card" }},
        "nationalIdCardDateOfIssue": {{ "type": "string", "description": "Date of issue of national ID card" }},
        "nationalIdCardDateOfExpiry": {{ "type": "string", "description": "Date of expiry of national ID card" }},
        "nationalIdCardCountryOfIssue": {{ "type": "string", "description": "Country of issue of national ID card" }},
        "nationalIdCardLastName": {{ "type": "string", "description": "Last name on national ID card" }},
        "nationalIdCardFirstName": {{ "type": "string", "description": "First name on national ID card" }}
      }}
    }} }},
    "individual.additionalAttributes.giveDetailsOfSupportingWitness": {{ "type": "array", "items": {{
      "type": "object",
      "properties": {{
        "homePhoneWitness": {{ "type": "string", "description": "Home phone of supporting witness" }},
        "numberOfYearsWitness": {{ "type": "string", "description": "Number of years known by supporting witness" }},
        "relationshipApplicantWitness": {{ "type": "string", "description": "Relationship of supporting witness to applicant" }},
        "occupationWitness": {{ "type": "string", "description": "Occupation of supporting witness" }},
        "addressWitness": {{ "type": "string", "description": "Address of supporting witness" }},
        "familyNameWitness": {{ "type": "string", "description": "Family name of supporting witness" }},
        "sexWitness": {{ "type": "string", "description": "Sex of supporting witness" }},
        "countryWitness": {{ "type": "string", "description": "Country of supporting witness" }},
        "dateOfBirthWitness": {{ "type": "string", "description": "Date of birth of supporting witness" }}
      }}
    }} }},
    "individual.additionalAttributes.inNominationForm": {{ "type": "string", "description": "In nomination form" }},
    "individual.additionalAttributes.otherNames": {{ "type": "string", "description": "Other names of the individual" }},
    "individual.additionalAttributes.otherIdentityDocument": {{ "type": "string", "description": "Other identity document" }},
    "individual.additionalAttributes.citizenOfAnyOtherCountry": {{ "type": "string", "description": "Whether citizen of any other country" }},
    "individual.additionalAttributes.passportIssuePlace": {{ "type": "string", "description": "Place of passport issue" }},
    "individual.additionalAttributes.nationalIdCard": {{ "type": "string", "description": "National ID card" }},
    "individual.lastName": {{ "type": "string", "description": "Last name of the individual" }},
    "individual.createdAt": {{ "type": "string", "description": "Creation date of the individual record" }},
    "individual.attachments": {{ "type": "array", "description": "Attachments related to the individual" }},
    "individual.passportNumber": {{ "type": "string", "description": "Passport number of the individual" }},
    "individual.invitationStatus": {{ "type": "string", "description": "Invitation status of the individual" }},
    "individual.firstName": {{ "type": "string", "description": "First name of the individual" }},
    "individual.passportNationality": {{ "type": "string", "description": "Passport nationality of the individual" }},
    "individual.updatedAt": {{ "type": "string", "description": "Last update date of the individual record" }},
    "individual.passportCountry": {{ "type": "string", "description": "Passport country of the individual" }},
    "individual.dateOfBirth": {{ "type": "string", "description": "Date of birth of the individual" }},
    "individual.age": {{ "type": "string", "description": "Age of the individual" }},
    "company.legalCompanyName": {{ "type": "string", "description": "Legal name of the company" }},
    "company.isGroupCompany": {{ "type": "string", "description": "Whether the company is a group company" }},
    "company.additionalAttributes.isTheApplicantOperatingInAustralia": {{ "type": "string", "description": "Whether the applicant is operating in Australia" }},
    "company.additionalAttributes.projectedPayroll": {{ "type": "string", "description": "Projected payroll of the company" }},
    "company.additionalAttributes.isTheapplicantTrustee": {{ "type": "string", "description": "Whether the applicant is a trustee" }},
    "company.additionalAttributes.addressOfResidence": {{ "type": "string", "description": "Address of residence" }},
    "company.additionalAttributes.doesTheApplicantDeclare": {{ "type": "string", "description": "Whether the applicant declares any information" }},
    "company.additionalAttributes.postalCodeSameAsResidentialAddress": {{ "type": "string", "description": "Whether postal code is same as residential address" }},
    "company.additionalAttributes.rangeOfAnnualTurnOver": {{ "type": "string", "description": "Range of annual turnover" }},
    "company.additionalAttributes.willTheApplicantPaySalary": {{ "type": "string", "description": "Whether the applicant will pay salary" }},
    "company.additionalAttributes.isTheApplicantRegistered": {{ "type": "string", "description": "Whether the applicant is registered" }},
    "company.additionalAttributes.annualTurnOver": {{ "type": "string", "description": "Annual turnover of the company" }},
    "company.additionalAttributes.countryOfResidence": {{ "type": "string", "description": "Country of residence" }},
    "company.additionalAttributes.familyNameBusiness": {{ "type": "string", "description": "Family name for business" }},
    "company.additionalAttributes.legalName": {{ "type": "string", "description": "Legal name of the business" }},
    "company.additionalAttributes.doesTheApplicantCurrentlyHaveALabourAgreement": {{ "type": "string", "description": "Whether the applicant currently has a labour agreement" }},
    "company.additionalAttributes.industryType": {{ "type": "string", "description": "Type of industry" }},
    "company.additionalAttributes.postalCodeOfResidence": {{ "type": "string", "description": "Postal code of residence" }},
    "company.additionalAttributes.istheApplicantCurrently": {{ "type": "string", "description": "Current status of the applicant" }},
    "company.additionalAttributes.suburbOfResidence": {{ "type": "string", "description": "Suburb of residence" }},
    "company.additionalAttributes.email": {{ "type": "string", "description": "Email address" }},
    "company.additionalAttributes.dateEstablished": {{ "type": "string", "description": "Date the company was established" }},
    "company.additionalAttributes.registrationDetailsNo": {{ "type": "array", "items": {{
      "type": "object",
      "properties": {{
        "businessRegistrationIdDetailsNo": {{ "type": "string", "description": "Business registration ID details number" }},
        "businessRegistrationDetailsNo": {{ "type": "string", "description": "Business registration details number" }}
      }}
    }} }},
    "company.additionalAttributes.givenNamesBusiness": {{ "type": "string", "description": "Given names for business" }},
    "company.additionalAttributes.ifTheStandardBusinessSponsorship": {{ "type": "string", "description": "If the standard business sponsorship" }},
    "company.additionalAttributes.doesTheApplicantWebpage": {{ "type": "string", "description": "Whether the applicant has a webpage" }},
    "company.additionalAttributes.positionInBusiness": {{ "type": "string", "description": "Position in business" }},
    "company.additionalAttributes.doestheApplicant": {{ "type": "string", "description": "Whether the applicant does something (context-specific)" }},
    "company.additionalAttributes.giveDescriptionBusiness": {{ "type": "string", "description": "Description of the business" }},
    "company.additionalAttributes.isThereAnyAdverseInformation": {{ "type": "string", "description": "Whether there is any adverse information" }},
    "company.additionalAttributes.doesTheApplicantHoldSponsorshipStatus": {{ "type": "string", "description": "Whether the applicant holds sponsorship status" }},
    "company.additionalAttributes.territoryOfResidence": {{ "type": "string", "description": "Territory of residence" }},
    "company.additionalAttributes.ownerDetails": {{ "type": "array", "items": {{
      "type": "object",
      "properties": {{
        "selectWhetherTheOwner": {{ "type": "string", "description": "Selection of owner type" }},
        "positionOwnersDirectors": {{ "type": "string", "description": "Position of owners/directors" }},
        "familyNameOwnersDirectors": {{ "type": "string", "description": "Family name of owners/directors" }},
        "givenNameOwnersDirectors": {{ "type": "string", "description": "Given name of owners/directors" }},
        "sexOwnersDirectors": {{ "type": "string", "description": "Sex of owners/directors" }},
        "dateOfBirthOwnersDirectors": {{ "type": "string", "description": "Date of birth of owners/directors" }}
      }}
    }} }},
    "company.additionalAttributes.mobilePhone": {{ "type": "string", "description": "Mobile phone number" }},
    "company.additionalAttributes.businessStructure": {{ "type": "string", "description": "Structure of the business" }},
    "company.additionalAttributes.isTheApplicantAustralianTaxationOffice": {{ "type": "string", "description": "Whether the applicant is registered with the Australian Taxation Office" }},
    "company.additionalAttributes.applicantTradingAustralia": {{ "type": "string", "description": "Whether the applicant is trading in Australia" }},
    "company.additionalAttributes.doesTheApplicantFranchise": {{ "type": "string", "description": "Whether the applicant is a franchise" }},
    "company.additionalAttributes.isTheApplicantAustralian": {{ "type": "string", "description": "Whether the applicant is Australian" }},
    "company.additionalAttributes.businessPhone": {{ "type": "string", "description": "Business phone number" }},
    "company.phone": {{ "type": "string", "description": "Company phone number" }},
    "company.companyNameInMigrationManager": {{ "type": "string", "description": "Company name in migration manager" }},
    "company.additionalAttributes.countryOfCommunication": {{ "type": "string", "description": "Country for communication" }},
    "company.additionalAttributes.addressOfCommunication": {{ "type": "string", "description": "Address for communication" }},
    "company.additionalAttributes.addressTwoOfCommunication": {{ "type": "string", "description": "Secondary address for communication" }},
    "company.additionalAttributes.suburbOfCommunication": {{ "type": "string", "description": "Suburb for communication" }},
    "company.additionalAttributes.stateOfCommunication": {{ "type": "string", "description": "State for communication" }},
    "company.additionalAttributes.postalCodeOfCommunication": {{ "type": "string", "description": "Postal code for communication" }},
    "matter.additionalAttributes.giveDetailsWhoWillPaysalary": {{ "type": "string", "description": "Details of who will pay the salary" }},
    "company.additionalAttributes.australianEmployees": {{ "type": "string", "description": "Number of Australian employees" }},
    "company.additionalAttributes.foreignEmployees": {{ "type": "string", "description": "Number of foreign employees" }},
    "formResponse": {{ 
      "type": "object",
      "properties": {{
        "matter": {{
          "type": "object",
          "properties": {{
            "additionalAttributes": {{
              "type": "object",
              "properties": {{
                "havereadandUnderstoodInformation": {{ "type": "string", "description": "Whether the applicant has read and understood the information" }},
                "haveprovidedcompleteAndCorrectInformation": {{ "type": "string", "description": "Whether the applicant has provided complete and correct information" }},
                "willinformtheDepartment": {{ "type": "string", "description": "Whether the applicant will inform the department of any changes" }},
                "havereadinformationContainedPrivacyNotice": {{ "type": "string", "description": "Whether the applicant has read the privacy notice" }},
                "understandthatTheDepartment": {{ "type": "string", "description": "Whether the applicant understands that the department may use the information" }},
                "organisationconsentsToSponsor": {{ "type": "string", "description": "Whether the organization consents to sponsor" }},
                "willComply": {{ "type": "string", "description": "Whether the applicant will comply with the requirements" }}
              }}
            }}
          }}
        }}
      }}
    }}
  }}

    Here is the information\n\n
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


pdf_path = "/Users/adityashedge/Downloads/application3.pdf"

chunks = read_and_split_pdf(pdf_path)

json_results = []

for i, chunk in enumerate(chunks):
    json_result = convert_to_json(chunk)
    json_results.append(json_result)
    print(f"Processed chunk {i+1}/{len(chunks)}")

combined_json = "[" + ",".join(json_results) + "]"

output_file = os.path.splitext(pdf_path)[0] + '_output.json'
with open(output_file, 'w') as f:
    f.write(combined_json)

print(f"JSON output saved to {output_file}")