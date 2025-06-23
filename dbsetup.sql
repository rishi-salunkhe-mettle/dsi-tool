CREATE DATABASE lava;

CREATE USER lava WITH PASSWORD 'password';

GRANT CONNECT ON DATABASE lava TO lava;

\c lava

CREATE TABLE predictions (patient_id VARCHAR(500), prediction_timestamp DATE, predicted_prob FLOAT, predicted_outcome INT);

GRANT USAGE ON SCHEMA public TO lava;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA public TO lava;
