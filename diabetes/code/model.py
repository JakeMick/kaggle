#! /usr/bin/env python
"""
Diabetus
"""
from os import path
import sqlite3

class processing():
    """ Helper class pulls data into numpy array.
    """
    def __init__(self):
        """ Defines general locations of project management.
        """
        current_path = path.dirname(path.abspath(__file__))
        parent_dir = path.dirname(current_path)
        self.data_dir = path.join(parent_dir, 'data')
        self.sub_dir = path.join(parent_dir, 'submissions')
        self.load_file_handles()

    def load_file_handles(self):
        """ Creates connection to database and a helper cursor.
        """
        self.conn = sqlite3.Connection(path.join(self.data_dir,'compData.db'))
        self.curr = sqlite3.Cursor(self.conn)

    def get_set(self, column_name, table):
        super_query = self.conn.execute("SELECT {0} from {1}".format(
            column_name, table))
        return super_query.fetchall()

    def get_column_names(self, table):
        self.curr.execute("PRAGMA table_info({0})".format(table))
        return [i[1] for i in self.curr.fetchall()]

    def get_datasets(self):
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

        for held_in_out in ['train','test']:
            for table in datanames:
                target_table = table.format(held_in_out)
                column_names = self.get_column_names(target_table)
                



