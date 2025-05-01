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

FHIR_SERVER_URL = ""
BEARER_TOKEN = ""

headers={
    "Authorization":f"Bearer {BEARER_TOKEN}",
    "Content-Type":"application/fhir+json"
}

RACE_MAPPING = {
   
    "American Indian or Alaska Native": {
        "system": "urn:oid:2.16.840.1.113883.6.238",
        "code": "1002-5",
        "display": "American Indian or Alaska Native"
    },
    "Asian": {
        "system": "urn:oid:2.16.840.1.113883.6.238",
        "code": "2028-9",
        "display": "Asian"
    }, 
    "Black": {
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
    "Heart Failure": "84114007",      # SNOMED CT for Heart failure
    "Diabetes": "44054006",           # Diabetes mellitus type 2
    "Hypertension": "38341003",       # Essential (primary) hypertension
    "CKD": "431855005"               # Chronic kidney disease
}

def generate_patient(patient_id, gender, race, birthdate):
    return Patient(
        id=patient_id,
        identifier=[
            Identifier(
                system="urn:ietf:rfc:3986", 
                value=patient_id
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

def generate_ckd_condition(patient_ref,actual_outcome,predicted_date_str):

    if str(actual_outcome).strip() ==  "1":
        predicted_date = datetime.strptime(predicted_date_str.strip(), "%Y-%m-%d %I:%M %p")
        onset_date = predicted_date + relativedelta(months=3)
        return Condition.construct(
            id=str(uuid.uuid4()),
            subject=Reference(reference=f"Patient/{patient_ref}"),
            code=CodeableConcept(
                coding=[Coding(system="http://snomed.info/sct", code="431855005", display="Chronic kidney disease stage 1 (disorder)")],
                text="Chronic kidney disease stage 1 (disorder)"
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

            obs1 = generate_observation("HbA1c", "http://loinc.org", "4548-4", "%", server_patient_id, row["HbA1c (%)"])
            obs2 = generate_observation("eGFR", "http://loinc.org", "62238-1", "mL/min/1.73m2", server_patient_id, row["eGFR"])
            obs3 = generate_observation("UACR", "http://loinc.org", "32294-1", "mg/g", server_patient_id, row["UACR"])
            post_resource(obs1)
            post_resource(obs2)
            post_resource(obs3)
            bundle.entry.extend([
                BundleEntry(resource= obs1),
                BundleEntry(resource= obs2),
                BundleEntry(resource= obs3)
                ])

            if row["Conditions"]:
                comorbidities_list = [c.strip() for c in row["Conditions"].split(",") if c.strip]
                for each in comorbidities_list:
                    if each.lower() not in seen_conditions[pid]:
                        print(each)
                        cond = generate_condition(each, server_patient_id)
                        post_resource(cond)
                        bundle.entry.append(BundleEntry(resource= cond))
                        seen_conditions[pid].add(each.lower())

            cond = generate_ckd_condition(server_patient_id,row['CKD_Actual_Outcome'],row['Prediction_Timestamp'])
            if cond:
                bundle.entry.append(BundleEntry(resource=cond))
                post_resource(cond)

    return bundle

def post_bundle_resources(bundle):
    for entry in bundle.entry:

        resource = entry.resource
        resource_type = resource.resource_type
       
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
    resource_type = resource.resource_type
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
    bundle = convert_csv_to_fhir_bundle("diabetes_prediction_trajectory_data 3.csv")
    
    bundle_json = bundle.json(indent =2)

    with open("output_bundle.json","w") as f:
        f.write(bundle_json)
    print("FHIR BUNDLE is created")

