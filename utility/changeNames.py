import glob
import os
import xml.dom.minidom as md

def main():

	filePath = ""
	endings = {"1", "1a", "2"}
	paths = {"Annotations", "frame"}
	files = glob.glob(filePath + "/*.xml")
	
	for end in endings:
		for path in paths:
			filePath = os.path.join("LuggageTagCombined", path + end)
			if(path == "Annotations"):
				files = glob.glob(filePath + "/*.xml")
				for filename, file in zip(os.listdir(filePath), files):
					f = os.path.join(filePath, filename)
					file = md.parse(file)

					# modifying the value of a tag(here "filename")
					file.getElementsByTagName("filename")[0].childNodes[0].nodeValue = str(
						file.getElementsByTagName("filename")[0].childNodes[0].nodeValue)[:-4] + "_" + end + ".PNG" 

					with open(f, "w") as fs:
						fs.write(file.toxml())
						fs.close()
						
					os.rename(os.path.join(filePath, filename), os.path.join(filePath, filename[:-4] + "_" + end + ".xml"))
					
			if(path == "frame"):
				for filename in os.listdir(filePath):
					#print(os.path.join(filePath, filename) + " " + filePath)
					os.rename(os.path.join(filePath, filename), os.path.join(filePath, filename[:-4] + "_" + end + ".PNG"))
			
if __name__ == "__main__":
    main();