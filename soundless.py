from modules.db_soundless import Db

db = Db()

data = db.getSleepSficiency()
db.insertEfficiencies(data[data["se"].notnull() & data["se"] > 0])

