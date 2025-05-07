import json

from modules.aws import AWS
from modules.chann_selector import ChannSelector
from modules.model import EEGModel
from modules.db import Db 

aws = AWS()
model = EEGModel()
db = Db()

def getTestData(folder, session, site):
    
    channels = ChannSelector().select(aws.loadEegChannelsTsv(folder, session, site))

    parser = aws.loadEegEdf(folder, session, site)
    annotations = aws.loadEegAnnotationsCsv(folder, session, site)

    parser.setAnottations(annotations)
    chunks = parser.crop(channels["name"].to_list())
    tags = parser.getTags()
    parser.purge()

    return chunks, tags


testInstances = db.getTest()

for folder, session, site in testInstances[0:1]:
    chunks, tags = getTestData(folder, session, site)

    data = []
    for chunk in list(map(lambda x: x.to_data_frame(), chunks)):
        data.append(chunk.drop(columns = "time"))

    print(model.evaluate(data, tags))
