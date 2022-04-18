import base64
import folium
from folium import IFrame
import random

# **** Current problem: the way that this is coded requires that the image file already exists upon instantiation
# Possible solution: Instead of calling addPoint() in the constructor, call it manually after saving the "plot" (image)

class Detection:

    staticId = 1

    def __init__(self, m):
        self.gps_controller = gps_controller
        self.fod_type = fod_type
        self.point = self.get_position()
        self.m = m
        self.id = self.staticId
        Detection.staticId += 1
        self.image = "detectionImages/" + str(self.id) + ".jpeg"
        # self.addPoint()
        print("Detection object created at coordinates: " + str(self.point))

    def addPoint(self):
        if self.point:  # If point is defined
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

    # Generate GPS coordinates
    # ToDo: will need to replace with device coordinates
    def get_position(self):
        lat = random.uniform(45.54968462694988, 45.55057682445797)
        long = random.uniform(-94.15178127548012, -94.15303386704261)
        return [lat, long]


