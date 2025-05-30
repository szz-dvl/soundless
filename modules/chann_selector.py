import pandas as pd

class MissingChannels(Exception):
    pass
    
class ChannSelector():
    def __init__(self):
        self.eeg_exclusive = [
            "C3-M2",
            "O1-M2",
        ]

        self.eeg_equivalences = [
            [
                "C4-M1",
                "C4-M2"
            ],
            [
                "E1-M1",
                "E1-M2"
            ],
            [
                "E2-M1",
                "E2-M2"
            ],
            [
                "F4-M1",
                "F4-M2",
                "F8-M1",
                "Fp2-M1",
                "Fp2-M2"
            ],
            [
                "O1-M1",
                "O2-M1",
                "O2-M2"
            ],
        ]

        self.mandatory = [
            "ABD",
            "C3-M2",
            "CHEST",
           # "E1-M2", (present in equivalences)
            "O1-M2",
            "IC",
            "SNORE",
           # "EKG", (present in equivalences)
            "AIRFLOW",
            "HR",
           # "LAT", (present in equivalences)
           # "RAT", (present in equivalences)
            "SaO2"
        ]

        self.equivalences = [
            [
                "C4-M1",
                "C4-M2"
            ],
            [
                "CHIN1-CHIN2",
                "CHIN1-CHIN3",
                "CHIN2-CHIN3",
                "CHIN3-CHIN1",
                "CHIN3-CHIN2",
                "Chin1-31",
                "Chin1-Chin2",
                "Chin1-Chin3",
                "Chin1-P3",
                "Chin1-P4",
                "Chin2-Chin3",
                "Chin3-Chin2"
            ],
            [
                "E1-M1",
                "E1-M2"
            ],
            [
                "E2-M1",
                "E2-M2"
            ],
            [
                "F4-M1",
                "F4-M2",
                "F8-M1",
                "Fp2-M1",
                "Fp2-M2"
            ],
            [
                "O1-M1",
                "O2-M1",
                "O2-M2"
            ],
            [
                "EKG",
                "EKG-E1"
            ],
            [
                "LAT",
                "LAT-E1"
            ],
            [
                "RAT",
                "RAT-E1"
            ]
        ]

    def __getOrderedChannels(self, channels: list) -> list:
        ordered = []
        for mandatory in self.mandatory:
            ordered.append(mandatory)

        for equivalence in self.equivalences:
            for chann in equivalence:
                if chann in channels:
                    ordered.append(chann)

        return ordered

    def select(self, rawChannels: pd.DataFrame) -> pd.DataFrame:
        mandatory = rawChannels[rawChannels["name"].isin(self.mandatory)]

        if len(mandatory) != len(self.mandatory):
            raise MissingChannels(f"Mandatory = {len(mandatory)}")
        
        for equivalence in self.equivalences:
            equivalent = rawChannels[rawChannels["name"].isin(equivalence)]
            if len(equivalent) != 1:
                raise MissingChannels(f"Equivalence = {len(equivalent)}")
            
            mandatory = pd.concat([mandatory, equivalent], sort = False)

        orderedIdx = []
        for chann in self.__getOrderedChannels(mandatory["name"].to_list()):
            orderedIdx.append(mandatory[mandatory["name"] == chann].index.values[0])
        
        return mandatory.reindex(orderedIdx)

    def __getOrderedChannelsEeg(self, channels: list) -> list:
        ordered = []
        for mandatory in self.eeg_exclusive:
            ordered.append(mandatory)

        for equivalence in self.eeg_equivalences:
            for chann in equivalence:
                if chann in channels:
                    ordered.append(chann)

        return ordered

    def selectEeg(self, rawChannels: pd.DataFrame) -> pd.DataFrame:
        mandatory = rawChannels[rawChannels["name"].isin(self.eeg_exclusive)]

        if len(mandatory) != len(self.eeg_exclusive):
            raise MissingChannels(f"Mandatory = {len(mandatory)}")
        
        for equivalence in self.eeg_equivalences:
            equivalent = rawChannels[rawChannels["name"].isin(equivalence)]
            if len(equivalent) != 1:
                raise MissingChannels(f"Equivalence = {len(equivalent)}")
            
            mandatory = pd.concat([mandatory, equivalent], sort = False)

        orderedIdx = []
        for chann in self.__getOrderedChannelsEeg(mandatory["name"].to_list()):
            orderedIdx.append(mandatory[mandatory["name"] == chann].index.values[0])
        
        return mandatory.reindex(orderedIdx)
    
    
