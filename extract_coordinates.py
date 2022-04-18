# This script is used for extracting the coordinates from the raw data of the GPS puck.

from pyembedded.gps_module.gps import GPS
import time
import sys

class GPS_Controller():

    last_coords = None

    def __init__(self) -> None:
        self.gps = self.get_gps_device()

    def get_gps_device(self):

        gps = None
        return None
        for i in range(30):
            # Port used for device, baud rate set for device
            portString = "Com" + str(i)
            print("Searching for GPS device on com port " + str(i))
            try:
                gps = GPS(port=portString, baud_rate=4800)
                print("Device found on port " + str(i))
                return gps
            except:
                print("device not found on port " + str(i))
        return None

    # The following function takes raw data (array) as argument, returns [longitude, latitude] as decimal degrees.
    # Raw data is in NMEA format, so we need to parse the numbers to get the true decimal-degree coordinates.
    # For example, 09349.7112 West is actually -93.82852 because the first digits (093) indicate degrees, 
    # while the rest of the number (49.7...) indicates the minutes.
    # West and South correspond to negative values for latitude and longitude, respectively.
    def transform_coordinates(self, raw_data):

        try:
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
        except:
            print("error while transforming coordinates")
            return None


    def get_raw_gps(self): # Returns raw data from GPS as an array
        # Port used for device, baud rate set for device
        #gps = GPS(port='Com4', baud_rate=4800)

        # Getting GPS data
        #print(gps.get_lat_long()) # Getting longitude and latitude of device
        #print(gps.get_no_of_satellites()) # Getting number of satellites used
        try:
            return self.gps.get_raw_data() # Raw GPS data
        except:
            print("unable to get gps coordinates")
        # time.sleep(.30)
            #sys.exit()

    def extract_coordinates(self):
        try:
            raw_data = self.get_raw_gps()
            transformed = self.transform_coordinates(raw_data)
            self.last_coords = transformed
            print("extract_coordinates: " + str(transformed))
            return transformed
        except:
            return [0, 0]

# Test case (only executes when this script is run directly, not when imported)
if __name__ == "__main__":

    gps_controller = GPS_Controller()
    # Sample reading from the GPS puck:
    #raw_data = ['$GPGGA', '200634.000', '4530.5608', 'N', '09349.7112', 'W', '1', '07', '1.1', '318.4', 'M', '-31.0', 'M', '', '0000*60']
    raw_data = gps_controller.get_raw_gps()
    print(raw_data)

    transformed = gps_controller.transform_coordinates(raw_data)
    print(transformed)

    extracted = gps_controller.extract_coordinates()
    print(extracted)