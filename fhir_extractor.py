import streamlit as st
import requests
import json
from fhir.resources.observation import Observation
from fhir.resources.condition import Condition
from fhir.resources.patient import Patient
from fhir.resources.encounter import Encounter
from typing import List
from datetime import datetime

def fetch_fhir_resource(base_url, resource_type, headers):
    """ Fetches FHIR resources from the provided FHIR server """
    url = f"{base_url}/{resource_type}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        return response.json().get("entry", [])  # Return list of resources
    else:
        st.error(f"Error {response.status_code}: Unable to fetch {resource_type} data.")
        return []

def extract_observation_value(observation_json, target_loinc):
    """ Extracts value from an Observation resource based on LOINC code """
    try:
        observation = Observation.parse_obj(observation_json)
        if observation.code and observation.code.coding:
            for coding in observation.code.coding:
                if coding.code == target_loinc:
                    return observation.valueQuantity.value if observation.valueQuantity else None
    except Exception as e:
        print(f"Failed to parse Observation: {e}")
    return None

def extract_conditions(conditions_json, condition_codes: List[str]):
    """ Extracts list of comorbidities from Condition resources """
    comorbidities = []
    for entry in conditions_json:
        try:
            condition = Condition.parse_obj(entry.get("resource", entry))
            if condition.code and condition.code.coding:
                for coding in condition.code.coding:
                    if coding.code in condition_codes:
                        comorbidities.append(coding.display or coding.code)
        except Exception as e:
            print(f"Failed to parse Condition: {e}")
    return comorbidities

def extract_patient_data(patient_json):
    try:
        patient = Patient.parse_obj(patient_json)
        gender = patient.gender
        birth_date = patient.birthDate
        patient_id = patient.id
        age = None
        if birth_date:
            birth_date_obj = datetime.strptime(str(birth_date), "%Y-%m-%d")
            age = int((datetime.now() - birth_date_obj).days / 365.25)

        race = None
        if patient.extension:
            for ext in patient.extension:
                if ext.url == "http://hl7.org/fhir/us/core/StructureDefinition/us-core-race":
                    for sub_ext in ext.extension:
                        if sub_ext.url == "text":
                            race = sub_ext.valueString

        return {"Patient ID": patient_id, "Gender": gender, "Age": age, "Race": race}
    except Exception as e:
        print(f"Failed to parse Patient: {e}")
        return {}

def extract_hospitalization(encounters_json):
    try:
        for entry in encounters_json:
            encounter = Encounter.parse_obj(entry.get("resource", entry))
            if encounter.status == "in-progress" and encounter.class_fhir and encounter.class_fhir.code == "IMP":
                return True
    except Exception as e:
        print(f"Failed to parse Encounter: {e}")
    return False

def main():
    st.title("FHIR Data Extractor")

    base_url = st.text_input("FHIR Server Base URL", placeholder="https://server.fire.ly")
    bearer_token = st.text_input("Bearer Token (Optional)", type="password")
    client_id = st.text_input("Client ID (Optional)")
    client_secret = st.text_input("Client Secret (Optional)", type="password")

    LOINC_CODES = {"HbA1c": "4548-4", "eGFR": "48642-3", "UACR": "9318-7"}
    CONDITION_CODES = ["38341003", "44054006"]  # Example: Hypertension, CKD

    results = {}

    if st.button("Fetch Data from Server"):
        if not base_url:
            st.error("FHIR Server Base URL are required!")
            return

        headers = {"Accept": "application/json"}
        if bearer_token:
            headers["Authorization"] = f"Bearer {bearer_token}"

        observations = fetch_fhir_resource(base_url, "Observation", headers)
        for name, loinc in LOINC_CODES.items():
            for entry in observations:
                value = extract_observation_value(entry.get("resource", {}), loinc)
                if value is not None:
                    results[name] = value
                    break

        conditions = fetch_fhir_resource(base_url, "Condition", headers)
        extracted_data = extract_conditions(conditions, CONDITION_CODES)
        if extracted_data:
            results["Comorbidities"] = extracted_data

        patients = fetch_fhir_resource(base_url, "Patient", headers)
        if patients:
            results.update(extract_patient_data(patients[0].get("resource", {})))

        encounters = fetch_fhir_resource(base_url, "Encounter", headers)
        results["Hospitalized"] = extract_hospitalization(encounters)

        if not results:
            st.warning("Required FHIR Data not found in Server Response")

    if results:
        st.subheader("FHIR Data extracted from Server")
        st.json(results)

    st.subheader("Or Upload FHIR Resource JSON")
    uploaded_obs = st.file_uploader("Upload Observation JSON", type=["json"])
    uploaded_cond = st.file_uploader("Upload Condition JSON", type=["json"])
    uploaded_pat = st.file_uploader("Upload Patient JSON", type=["json"])
    uploaded_enc = st.file_uploader("Upload Encounter JSON", type=["json"])

    observation_results = {}
    if uploaded_obs:
        obs_data = json.load(uploaded_obs)
        if isinstance(obs_data, dict) and "entry" in obs_data:
            entries = obs_data["entry"]
        elif isinstance(obs_data, list):
            entries = obs_data
        else:
            entries = [obs_data]

        for name, loinc in LOINC_CODES.items():
            for entry in entries:
                value = extract_observation_value(entry.get("resource", entry), loinc)
                if value is not None:
                    observation_results[name] = value
                    break

    if observation_results:
        st.subheader("FHIR Data extracted from Observation FHIR resource")
        st.json(observation_results)

    condition_results = {}
    if uploaded_cond:
        cond_data = json.load(uploaded_cond)
        if isinstance(cond_data, dict) and "entry" in cond_data:
            cond_entries = cond_data["entry"]
        elif isinstance(cond_data, list):
            cond_entries = cond_data
        else:
            cond_entries = [cond_data]

        condition_results["Comorbidities"] = extract_conditions(cond_entries, CONDITION_CODES)

    if condition_results:
        st.subheader("FHIR Data extracted from Condition FHIR Resource")
        st.json(condition_results)

    patient_results = {}
    if uploaded_pat:
        pat_data = json.load(uploaded_pat)
        patient_results.update(extract_patient_data(pat_data))

    if patient_results:
        st.subheader("FHIR Data extracted from Patient FHIR Resource")
        st.json(patient_results)

    encounter_result = {}
    if uploaded_enc:
        enc_data = json.load(uploaded_enc)
        if isinstance(enc_data, dict) and "entry" in enc_data:
            enc_entries = enc_data["entry"]
        elif isinstance(enc_data, list):
            enc_entries = enc_data
        else:
            enc_entries = [enc_data]

        encounter_result["Hospitalized"] = extract_hospitalization(enc_entries)

    if encounter_result:
        st.subheader("FHIR Data extracted from Encounter FHIR Resource")
        st.json(encounter_result)

if __name__ == "__main__":
    main()
