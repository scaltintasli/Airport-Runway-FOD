import glob
import os
import xml.dom.minidom as md

def main():

    filePath = os.path.join("Test")
    files = glob.glob(filePath + "/*.xml")

    for filename, file in zip(os.listdir(filePath), files):
        f = os.path.join(filePath, filename)
        file = md.parse(file)

        # modifying the value of a tag(here "age")
        file.getElementsByTagName("object")[0].getElementsByTagName("name")[0].childNodes[0].nodeValue = "object"

        with open(f, "w") as fs:
            fs.write(file.toxml())
            fs.close()

if __name__ == "__main__":
    main();
