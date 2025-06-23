import csv
import uuid
import json
import requests
from dateutil.relativedelta import relativedelta
from fhir.resources.bundle import Bundle,BundleEntry
from fhir.resources.patient import Patient
from fhir.resources.observation import Observation
from fhir.resources.humanname import HumanName
from fhir.resources.condition import Condition
from fhir.resources.extension import Extension
from fhir.resources.reference import Reference
from datetime import datetime
from fhir.resources.codeableconcept import CodeableConcept
from fhir.resources.coding import Coding
from fhir.resources.identifier import Identifier
from fhir.resources.bundle import Bundle

FHIR_SERVER_URL = "https://bfee16.devhcp.com/fhir"
BEARER_TOKEN = "eyJ0eXAiOiJKV1QiLCJhbGciOiJSUzI1NiJ9.eyJzY29wZSI6IlBhdGllbnQvKi5yZWFkIFBhdGllbnQvKi53cml0ZSIsInVzZXJfaWQiOjM4NywiY3VycmVudF9jb21wYW55Ijo2LCJpYXQiOjE3NDcyNTQyODQsImV4cCI6MTc0NzI1Nzg4NH0.hw9XffH3m1t1fbOTK9OfOyPhSp102WzasEtWN0kji5rX9juDorJ_at4R8DdACz8Qpnbb3Tgf2PqwxJVfFacrBMFoL0FU5yIAubB-D4WL6E4lENYOzoKG2EyfZwyNAV7ZCV0-J_5egj_Ii-Zd1nbUNFcsndmU7thyLqOhzYD3GSi7WrXCli_pCfM7FP3UtsMqNH6NQS1oDwKT5sY6Y_GONZUiJ3U9B7npcFExWo4-zbMxHP4tGYSv12igWVuj1cm4CTeCfkUQ3t4pSDq9CXIGmBiiLRyW0wQtNzmnqjKLQ5MorZLZeKXvfvXrycM3H8t3C0OLi_4ttxvPbOylk3Gnag"

headers={
    "Authorization":f"Bearer {BEARER_TOKEN}",
    "Content-Type":"application/fhir+json"
}

RACE_MAPPING = {
   
    "African American": {
        "system": "urn:oid:2.16.840.1.113883.6.238",
        "code": "2054-5",
        "display": "Black"
    },
    "White": {
        "system": "urn:oid:2.16.840.1.113883.6.238",
        "code": "2106-3",
        "display": "White"
    },
    "Other": {
        "system": "http://terminology.hl7.org/CodeSystem/v3-NullFlavo",
        "code": "UNK",
        "display": "Mixed"
    }
}

condition_snomed_map = {
    "Smoker": "449868002",      # SNOMED CT for Smoker
    "Non-Smoker": "266919005",  # SNOMED CT for Non-Smoker
    "Diabetic": "44054006",           # Diabetes mellitus type 2
    "Non-Diabetic": "410516002",      # Non-diabetic
    "On Treatment": "302192008",       
    "Not on Treatment": "170588009"    
}

def generate_patient(patient_id, gender, race, birthdate):
    return Patient(
        id=patient_id,
        identifier=[
            Identifier(
                system="urn:ietf:rfc:3986", 
                value=patient_id
            ),
            Identifier(
                system="lava", 
                value="lava_test"
            )
        ],

        gender=gender.lower(),
        name=[HumanName(
            use="official",
            given=[patient_id],
            family=patient_id
            )],
        birthDate=birthdate,
        extension=[{
            "url": "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race",
            "extension": create_race_extension(race)
        }]
    )

def create_race_extension(race):
    if race not in RACE_MAPPING:
        race = "Other"

    race = RACE_MAPPING[race]

    race_sub_extensions = [
        Extension.construct(
            url="ombCategory",
            valueCoding=Coding.construct(
                system=race['system'],
                code=race['code'],
                display=race['display']
            )
        ),
        Extension.construct(
            url="text",
            valueString=race['display']
        )
    ]
    return race_sub_extensions

def generate_observation(code_text, code_system, code_value, unit, patient_ref, value):
    return Observation(
        id=str(uuid.uuid4()),
        status="final",
        code=CodeableConcept(
            coding=[Coding(system=code_system, code=code_value, display=code_text)],
            text=code_text
        ),
        subject=Reference(reference=f"Patient/{patient_ref}"),
        valueQuantity={
            "value": float(value),
            "unit": unit
        },
        # effectiveDateTime=datetime.fromisoformat("2024-04-01")
    )

def generate_condition(cond_text, patient_ref):
    if not cond_text:
        return None
    
    code =condition_snomed_map.get(cond_text)

    if not code:
        return None
    
    return Condition(
        id=str(uuid.uuid4()),
        subject=Reference(reference=f"Patient/{patient_ref}"),
        code=CodeableConcept(
            coding=[Coding(system="http://snomed.info/sct", code=code, display=cond_text)],
            text=cond_text
        ),
        clinicalStatus={"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]}
    )

def generate_ascvd_condition(patient_ref,actual_outcome,predicted_date_str):

    if str(actual_outcome).strip() ==  "1":
        predicted_date = datetime.strptime(predicted_date_str.strip(), "%Y-%m-%d %I:%M %p")
        onset_date = predicted_date + relativedelta(months=3)
        return Condition.construct(
            id=str(uuid.uuid4()),
            subject=Reference(reference=f"Patient/{patient_ref}"),
            code=CodeableConcept(
                coding=[
                    Coding(
                        system="http://snomed.info/sct",
                        code="429493009",
                        display="Atherosclerotic cardiovascular disease (disorder)"
                    )
                ],
                text="Atherosclerotic cardiovascular disease (disorder)"
            ),
            clinicalStatus={"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-clinical", "code": "active"}]},
            verificationStatus={"coding": [{"system": "http://terminology.hl7.org/CodeSystem/condition-ver-status", "code": "confirmed"}]},
            onsetDateTime=onset_date.isoformat()
        )
    return None

def convert_csv_to_fhir_bundle(csv_path):
    bundle = Bundle(type="collection", entry=[])
    seen_patients={}
    seen_conditions={}

    with open(csv_path, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            pid = row["Patient_ID"]

            if pid not in seen_patients:
                patient = generate_patient(pid, row["Gender"], row["Race"],row['Birthdate'])
                seen_patients[pid] = patient
                seen_conditions[pid] = set()
                bundle.entry.append(BundleEntry(resource= patient))
                server_patient_id = post_resource(patient)

            obs1 = generate_observation("Total Cholesterol", "http://loinc.org", "2093-3", "mg/dL", server_patient_id, row["Total_Cholesterol"])
            obs2 = generate_observation("HDL Cholesterol", "http://loinc.org", "2085-9", "mg/dL", server_patient_id, row["HDL_Cholesterol"])
            obs3 = generate_observation("Systolic Blood Pressure", "http://loinc.org", "8480-6", "mmHg", server_patient_id, row["Systolic_Blood_Pressure"])
            post_resource(obs1)
            post_resource(obs2)
            post_resource(obs3)
            bundle.entry.extend([
                BundleEntry(resource= obs1),
                BundleEntry(resource= obs2),
                BundleEntry(resource= obs3)
                ])

            if row["Diabetes_Status"] and row["Current_Smoking_Status"] and row["Hypertension_Treatment_Status"]:
                for each in [row["Diabetes_Status"], row["Current_Smoking_Status"], row["Hypertension_Treatment_Status"]]:
                    if each.lower() not in seen_conditions[pid]:
                        print(each)
                        cond = generate_condition(each, server_patient_id)
                        post_resource(cond)
                        bundle.entry.append(BundleEntry(resource= cond))
                        seen_conditions[pid].add(each.lower())

            cond = generate_ascvd_condition(server_patient_id,row['ASCVD_Actual_Outcome'],row['Prediction_Timestamp'])
            if cond:
                bundle.entry.append(BundleEntry(resource=cond))
                post_resource(cond)

    return bundle

def post_bundle_resources(bundle):
    for entry in bundle.entry:

        resource = entry.resource
        resource_type = resource.__resource_type__
       
        url = f"{FHIR_SERVER_URL}/{resource_type}"

        resource_json =  json.loads(resource.json())
        response = requests.post(url,headers=headers,json= resource_json)

        if response.status_code in [200,201]:
            print("Created successfully")
        else:
            print("Failed")
            print(response.text)

def post_resource(resource):

    if not resource:
        return None 
    resource_type = resource.__resource_type__
    url = f"{FHIR_SERVER_URL}/{resource_type}"

    resource_json =  json.loads(resource.json())
    response = requests.post(url,headers=headers,json= resource_json)

    if response.status_code in [200,201]:
        print("Created successfully")
        print(response.json())
        return response.json().get("id")
    else:
        print("Failed")
        print(response.text)

if __name__ == "__main__":
    bundle = convert_csv_to_fhir_bundle("ASCVD_prediction_categorical_data.csv")
    
    bundle_json = bundle.json(indent =2)

    with open("output_bundle.json","w") as f:
        f.write(bundle_json)
    print("FHIR BUNDLE is created")

