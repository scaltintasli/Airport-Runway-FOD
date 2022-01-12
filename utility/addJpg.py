import glob
import os
import xml.dom.minidom as md

def main():

    filePath = os.path.join("NewScrewLabels", "Annotations")
    files = glob.glob(filePath + "/*.xml")

    for filename, file in zip(os.listdir(filePath), files):
        f = os.path.join(filePath, filename)
        file = md.parse(file)

        # modifying the value of a tag(here "age")
        file.getElementsByTagName("filename")[0].childNodes[0].nodeValue = str(
            file.getElementsByTagName("filename")[0].childNodes[0].nodeValue) + ".jpg"

        with open(f, "w") as fs:
            fs.write(file.toxml())
            fs.close()

if __name__ == "__main__":
    main();