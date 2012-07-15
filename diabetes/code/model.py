#! /usr/bin/env python
"""
Diabetus
"""
from os import path
import sqlite3
from collections import Counter

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

    def get_tables_column(self, table, column_name):
        try:
            super_query = self.conn.execute("SELECT {0} from {1}".format(
                column_name, table))
        except:
            print(column_name, table)
            raise()

        return super_query.fetchall()

    def get_tables_column_where(self, column_name, table, target, stuff):
        super_query = self.conn.execute("SELECT {0} from {1} where {2}={3}".format(
            column_name, table, target, stuff))
        return super_query.fetchall()

    def get_column_names(self, table):
        self.curr.execute("PRAGMA table_info({0})".format(table))
        return self.curr.fetchall()

    def most_unique(list_o_stuff, size):
        c = Counter(list_o_stuff)
        common_stuff = []
        for stuff, count in  c.most_common():
            if float(count)/float(size) <= .01:
                break
            else:
                common_stuff.append(stuff)
        return common_stuff

    def test_if_fake_int(self, list_o_stuff):
        total_len = 0.0
        numeric_len = 0.0
        for v in list_o_stuff:
            value = v[0]
            print(value)
            if value == None:
                continue
            else:
                total_len += float(len(value))
                numeric_len += sum([1.0 for i in value if i.isdigit()])
        print(numeric_len, total_len)
        return numeric_len/total_len >= 0.90

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

        for table in datanames:
            train_table = table.format('training')
            test_table = table.format('testing')
            columns = self.get_column_names(train_table)
            for col in columns:
                if col[2] == (u'INT' or u'REAL'):
                    pass
                else:
                    values = self.get_tables_column(train_table, col[1])
                    print(col, self.test_if_fake_int(values))





