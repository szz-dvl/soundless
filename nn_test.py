import json

from modules.aws import AWS
from modules.chann_selector import ChannSelector
from modules.model import EEGModel

aws = AWS()
model = EEGModel()

def getTestData(folder, session, site):
    
    channels = ChannSelector().select(aws.loadEegChannelsTsv(folder, session, site))

    parser = aws.loadEegEdf(folder, session, site)
    annotations = aws.loadEegAnnotationsCsv(folder, session, site)
    parser.setAnottations(annotations)
    chunks = parser.crop(channels["name"].to_list())
    tags = parser.getTags()
    parser.purge()

    return chunks, tags


with open("out/test_instances.json", "r") as jsonFile:
    testInstances = json.load(jsonFile)

    for tesInstance in testInstances[0:1]:
        chunks, tags = getTestData(tesInstance["folder"], tesInstance["session"], tesInstance["site"])
        print(model.evaluate(list(map(lambda x: x.get_data(), chunks)), tags))
