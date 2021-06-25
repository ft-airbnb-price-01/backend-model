""" This module holds the db queries for Postgresql """

# This query will create a new air_b table if one doesn't exist
CREATE_AIR_TABLE= """
CREATE TABLE IF NOT EXISTS air_b 
    (index SERIAL PRIMARY KEY,
    property_type INTEGER,
    room_type INTEGER,
    accommodates INTEGER,
    bathrooms INTEGER,
    bed_type INTEGER,
    cancellation_policy INTEGER,
    cleaning_fee INTEGER,
    city INTEGER,
    host_identity_verified INTEGER,
    host_since INTEGER,
    instant_bookable INTEGER,
    review_scores_rating INTEGER,
    zipcode INTEGER,
    bedrooms INTEGER,
    beds INTEGER,
    price REAL)
"""

# This query will select everything from the air_b table
GET_AIR_B= """
SELECT * 
FROM air_b; 
"""