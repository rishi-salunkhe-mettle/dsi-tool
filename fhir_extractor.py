import streamlit as st
import requests
import json
from fhir.resources.observation import Observation
from fhir.resources.condition import Condition
from typing import List

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

def main():
    st.title("FHIR Data Extractor")

    # User input for FHIR Server details
    base_url = st.text_input("FHIR Server Base URL", placeholder="https://server.fire.ly")
    bearer_token = st.text_input("Bearer Token (Optional)", type="password")
    client_id = st.text_input("Client ID (Optional)")
    client_secret = st.text_input("Client Secret (Optional)", type="password")
    
    # LOINC and SNOMED codes
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

        # Fetch Observations
        observations = fetch_fhir_resource(base_url, "Observation", headers)
        for name, loinc in LOINC_CODES.items():
            for entry in observations:
                value = extract_observation_value(entry.get("resource", {}), loinc)
                if value is not None:
                    results[name] = value
                    break

        # Fetch Conditions
        conditions = fetch_fhir_resource(base_url, "Condition", headers)
        extracted_data = extract_conditions(conditions, CONDITION_CODES)
        
        if extracted_data:
            results["Comorbidities"] = extracted_data

        if not results:
            st.warning("Required FHIR Data not found in Server Response")

    if results:
        st.subheader("FHIR Data extracted from Server")
        st.json(results)

    # File upload
    st.subheader("Or Upload FHIR Resource JSON")
    uploaded_obs = st.file_uploader("Upload Observation JSON", type=["json"])
    uploaded_cond = st.file_uploader("Upload Condition JSON", type=["json"])

    observation_results = {}

    if uploaded_obs:
        obs_data = json.load(uploaded_obs)
        # Normalize to list
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
        # Normalize to list
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

if __name__ == "__main__":
    main()
