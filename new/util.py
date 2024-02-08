def calculateTimeDependents(a, b, t):
    return round((a[0] - b[0]) / t, 3)


def averageShape(matrix):
    zeroTwo = []
    oneThree = []
    for row in matrix:
        zeroTwo.append(row[2] - row[0])
        oneThree.append(row[3] - row[1])
    width = average(zeroTwo)
    height = average(oneThree)
    return width/height


def average(list):
    return sum(list) / len(list)


def sortHighest(list):
    return sorted(list, key=lambda x: x[2], reverse=True)


# define all setup functions:

def displacement():

def shape():

def boundary():

def featureEmbedding():

    def getColour(self, detection):
        width = detection[2] - detection[0]
        height = detection[3] - detection[1]
        b, g, r = self.img[int((detection[1] + height / 2)), int((detection[0] + width / 2))]
        color = sRGBColor(r, g, b)
        return convert_color(color, LabColor)

    def getDistance(self, detection):
        refHeight = 194  # this is the height of the model pulled from monodepth2, to ensure the correct standard is used (could also change to percentage of screen but either works)
        scale = refHeight / self.imgH # scale to shift the measured detection height to the standard -------------------(ensure this is screen hieght)
        cls = round(detection[5]) # gets the class, so the correst reference equation is used

        # current distance
        height = detection[3] - detection[1] # measured detection height and then scale
        height = height * scale

        equ = referenceValues[cls]  # reference numerator, reference power
        Dist = round(equ[0] * height ** equ[1], 3) # estimates distance from the above set of equations

        return Dist

    def load_classes(self):
        # Loads *.names file at 'path'
        with open(self.path, 'r') as f:
            names = f.read().split('\n')
        return list(filter(None, names))  # filter removes empty strings (such as lastline)

    # def updateTracks(self):

# ...
