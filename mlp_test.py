from concurrent.futures import ThreadPoolExecutor, as_completed

from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, classification_report, confusion_matrix
from modules.aws import AWS
from modules.chann_selector import ChannSelector
from modules.mlp import MLPEegModel
from modules.db_mlp import Db 

aws = AWS()
model = MLPEegModel()
db = Db()

def parseData(data): 

    folder, session, site = data

    channels = ChannSelector().selectEeg(aws.loadEegChannelsTsv(folder, session, site))
    
    parser = aws.loadEegEdf(folder, session, site)
    annotations = aws.loadEegAnnotationsCsv(folder, session, site)

    parser.setAnottations(annotations)
    features, labels = parser.featuresPerEvent(channels)
    parser.purge()

    return features, labels
#124
def populateTest(): 

    full = db.sampleNum("test_data") != 0

    if full:
        print("Skipping tests population")
        return
    
    page = 0
    for data in db.paginateTest():
        with ThreadPoolExecutor(max_workers=len(data)) as executor:
            futureToRow = {executor.submit(parseData, row): row for row in data}
            for future in as_completed(futureToRow):
                row = futureToRow[future]
                try:
                    
                    features, labels = future.result()
                    db.insertFeatures(features, labels, "test_data")
                        
                except Exception as ex:
                    print(f"Exception %s: %s"%(row[0], ex))
                    pass

        print(f"PAGE {page} DONE!")
        page += 1

populateTest()
y_pred = []

for categorical in model.predict():
    y_pred.append(np.argmax(categorical))

y_test = db.getTestLabels()

cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3], normalize='true')
cr = classification_report(y_test, y_pred, labels=[0,1,2,3], target_names=[
            "Sleep_stage_R",
            "Sleep_stage_W",
            "Sleep_stage_N1/N2",
            "Sleep_stage_N3",
        ]
)
print(cr)
ConfusionMatrixDisplay(cm, display_labels=[
            "Sleep_stage_R",
            "Sleep_stage_W",
            "Sleep_stage_N1/N2",
            "Sleep_stage_N3",
            ]).plot()
plt.show()