#! /usr/bin/python
"""
Diabetus
"""
from os import path
import sqlite3

class processing():
    """Returns a class object with a dict of file handle.

    >>>herp = processing()
    """
    def __init__(self):
        current_path = path.dirname(path.abspath(__file__))
        parent_dir = path.dirname(current_path)
        self.data_dir = path.join(parent_dir, 'data')
        self.sub_dir = path.join(parent_dir, 'submissions')
        self.load_file_handles()

    def load_file_handles(self):
        self.database = sqlite3.Connection(path.join(self.data_dir,'compData.db'))

    def get_datasets(self):
        self.hash_mappings = {}
        datanames = [
        '{0}_allMeds',
        '{0}_allergy',
        '{0}_diagnosis',
#       '{0}_immunization',
#       '{0}_labObservation',
#       '{0}_labPanel',
        '{0}_labResult',
        '{0}_labs',
        '{0}_medication',
        '{0}_patient',
        '{0}_patientCondition',
        '{0}_patientSmokingStatus',
        '{0}_patientTranscript',
        '{0}_prescription',
        '{0}_smoke',
        '{0}_transcript',
#       '{0}_transcriptAllergy',
#       '{0}_transcriptDiagnosis',
#       '{0}_transcriptMedication'
        ]
        def get_set(train_or_test, table):
            super_query = self.database.execute("SELECT * from {0}".format(
                table.format(train_or_test)))
            return super_query.fetchall()

