# This script is used for extracting the coordinates from the raw data of the GPS puck.

# Takes raw data (array) as argument, returns [longitude, latitude] as decimal degrees.
# Raw data is in NMEA format, so we need to parse the numbers to get the true decimal-degree coordinates.
# For example, 09349.7112 West is actually -93.82852 because the first digits (093) indicate degrees, 
# while the rest of the number (49.7...) indicates the minutes (plus the remainder).
# West and South correspond to negative values for latitude and longitude, respectively.
def extract_coordinates(raw_data):

    ### Processing Latitude ###

    # Parse raw data to get latitude as degrees, minutes, seconds, and direction (West/East).
    lat = raw_data[2]
    lat_dir = raw_data[3]
    lat_deg = float(lat[0:2])
    lat_min = float(lat[2:4])
    lat_sec = float(lat[4:])
    # Converting remainder min to sec (might be unnecessary)
    lat_sec_conv = float(lat_sec) * 60.0


    ### Processing Longitude ###

    # Parse raw data, similar to previous calculations
    long = raw_data[4]
    long_dir = raw_data[5]
    long_deg = float(long[0:3])
    long_min = float(long[3:5])
    long_sec = float(long[5:])
    long_sec_conv = float(long_sec) * 60.0

    ### Converting to decimal degrees.
    # Now that we have the real deg/min/sec, convert to decimal degree value.
    # Dec. Deg. will be used for adding markers to the map.

    # Lat
    lat_dec_deg = lat_deg + (lat_min/60.0) + (lat_sec_conv/3600.0)
    if lat_dir == 'S':
        lat_dec_deg *= -1

    # Long
    long_dec_deg = long_deg + (long_min/60.0) + (long_sec_conv/3600.0)
    if long_dir == 'W':
        long_dec_deg *= -1

    lat_final = lat_dec_deg
    long_final = long_dec_deg

    #print(lat_final, long_final)

    # Returning the latitude and longitude to be used for placing point on map
    return [lat_final, long_final]

# Test case (only executes when this script is run directly, not when imported)
if __name__ == "__main__":

    # Sample reading from the GPS puck:
    raw_data = ['$GPGGA', '200634.000', '4530.5608', 'N', '09349.7112', 'W', '1', '07', '1.1', '318.4', 'M', '-31.0', 'M', '', '0000*60']
    
    print(extract_coordinates(raw_data))