import base64
import folium
from folium import IFrame
import random
from extract_coordinates import *


class Detection:

    staticId = 1

    def __init__(self, fod_type, m, gps_controller):
        self.gps_controller = gps_controller
        self.fod_type = fod_type
        self.point = self.get_position()
        self.m = m
        self.id = self.staticId
        Detection.staticId += 1
        self.image = "detectionImages/" + str(self.id) + ".jpeg"
        #self.addPoint()
        print("Detection object created at coordinates: " + str(self.point))

    def addPoint(self):
        if self.point: # If point is defined
            width = 500
            height = 500
            encoded = base64.b64encode(open(self.image, 'rb').read())
            html = '<img src="data:image/png;base64, {}" style="height:100%;width:100%;">'.format
            iframe = IFrame(html(encoded.decode('UTF-8'), width, height))
            popup = folium.Popup(iframe, min_width=1000, max_width=2650)
            folium.Marker(self.point, popup=popup).add_to(self.m)
            self.m.save("map.html")

    # For simulating fake GPS coordinates (ECC parking lot)
    def get_position_simulated(self):
        lat = random.uniform(45.54968462694988, 45.55057682445797)
        long = random.uniform(-94.15178127548012, -94.15303386704261)
        return [lat, long]

    # For getting real coordinates from GPS device
    def get_position(self):
        try:
            #coords = self.gps_controller.extract_coordinates()
            coords = self.gps_controller.last_coords
            return coords
        except:
            return None

# Testing (only executes if this file is run directly)
if __name__ == "__main__":

    gps_controller = GPS_Controller()
    m = folium.Map(location=[45.550120, -94.152411], zoom_start=20)

    tile = folium.TileLayer(
            tiles = 'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
            attr = 'Esri',
            name = 'Esri Satellite',
            overlay = False,
            control = True
        ).add_to(m)

    det1 = Detection("wood", m, gps_controller)
    det2 = Detection("wood", m, gps_controller)
    det3 = Detection("wood", m, gps_controller)
    print(det1.id)
    print(det2.id)
    print(det3.id)
    m.save("map.html")

    print(det1.get_position())