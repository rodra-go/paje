CREATE SCHEMA raw;

CREATE TABLE raw.individual_household_power_consumption (
    _id SERIAL PRIMARY KEY,
    Date TEXT,
    Time TEXT,
    Global_active_power TEXT,
    Global_reactive_power TEXT,
    Voltage TEXT,
    Global_intensity TEXT,
    Sub_metering_1 TEXT,
    Sub_metering_2 TEXT,
    Sub_metering_3 TEXT
);