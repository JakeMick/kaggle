#! /usr/bin/env python
"""
Diabetus
"""
from os import path
import sqlite3
import re
from collections import Counter
import numpy as np

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
        self.ordinal_threshold = 5

    def strip_bad_characters(self,string):
        no_unicode = ''.join((c for c in string if 0 < ord(c) < 127))
        no_html = re.sub('\&.*\;', '', no_unicode)
        return no_html

    def load_file_handles(self):
        """ Creates connection to database and a helper cursor.
        """
        self.conn = sqlite3.Connection(path.join(self.data_dir,'compData.db'))
        self.curr = sqlite3.Cursor(self.conn)

    def get_tables_column(self, table, column_name):
        super_query = self.conn.execute("SELECT {0} from {1}".format(
            column_name, table))

        return super_query.fetchall()

    def get_tables_column_where(self, column_name, table, target, stuff):
        super_query = self.conn.execute("SELECT {0} from {1} where {2}={3}".format(
            column_name, table, target, stuff))
        return super_query.fetchall()

    def get_column_names(self, table):
        self.curr.execute("PRAGMA table_info({0})".format(table))
        return self.curr.fetchall()

    def most_unique(self, list_o_stuff, size):
        c = Counter(list_o_stuff)
        common_stuff = []
        for stuff, count in  c.most_common():
            if float(count)/float(size) <= .01:
                break
            else:
                common_stuff.append(stuff)
        return common_stuff

    def test_if_fake_int(self, list_o_stuff):
        for v in list_o_stuff:
            value = v[0]
            if type(value) == int or type(value) == float:
                return True
            if value == None:
                continue

    def convert_to_number_list(self, list):
        out = []
        for v in list:
            value = v[0]
            if value == None:
                out.append(np.nan)
            else:
                numba = ''.join([s for s in value if s.isdigit()])
                if numba == '':
                    out.append(np.nan)
                else:
                    out.append(float(numba))

    def is_list_unique(self,list):
        """Determines if the first elements in a list of tuples is unique.
        """
        s = set([])
        for v in list:
            value = v[0]
            if value not in s:
                s.add(value)
            else:
                return False
        return True

    def unique_second_element(self,list):
        """ Returns a list of the unique seconds elements in a list of tuples.
        """
        return set([i[1] for i in list])

    def get_tables_column_patient(self, table, column_name):
        super_query = self.conn.execute("SELECT PatientGuid, {0} from {1}".format(
            column_name, table))
        return [s for s in super_query.fetchall() if s[1] != None]

    def add_subdict(self, dic, sub_name):
        if sub_name not in dic:
            dic[sub_name] = {}
        return dic

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

        self.train_patient_info = {}
        self.test_patient_info = {}
        definitely_numeric = set(['MedicationStrength'])
        for table in datanames:
            train_table = table.format('training')
            test_table = table.format('test')
            columns = self.get_column_names(train_table)
            for col in columns:
                col = list(col)
                col[1] = self.strip_bad_characters(col[1])
                print(col)
                # FIXME: formatting issues?
                if col[1][-2] == ':':
                    pass
                else:
                    try:
                        train_values = self.get_tables_column(
                                train_table, col[1])
                        # Used to lazily pass over tables not in test
                        test_values = self.get_tables_column(
                                test_table, col[1])
                    except:
                        print("Column {0} failed on table {1}".format(col[1],table))
                        continue
                    if self.test_if_fake_int(train_values):
                        train_both = self.get_tables_column_patient(
                                train_table, col[1])
                        test_both = self.get_tables_column_patient(
                                test_table, col[1])
                        if self.is_list_unique(train_both):
                            elems = self.unique_second_element(train_both)
                            if len(elems) <= self.ordinal_threshold:
                                for e in elems:
                                    for train, test in zip(train_both, test_both):
                                        # One patientID per table
                                        # Categorical/ordinal response
                                        self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                        self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])

                                        self.train_patient_info[train[0]][str(col[1])+'_'+str(e)] = int(train[1] == e)
                                        self.test_patient_info[test[0]][str(col[1])+'_'+str(e)] = int(test[1] == e)
                            else:
                                for train, test in zip(train_both, test_both):
                                        # One patientID per table
                                        # Lots of numerical responses
                                        self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                        self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])

                                        self.train_patient_info[train[0]][str(col[1])] = train[1]
                                        self.test_patient_info[test[0]][str(col[1])] = test[0]
                        else:
                            for train, test in zip(train_both, test_both):
                            # Count table occurences
                                self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])
 
                                resp = str(col[1]+'_'+'count')
                                if resp not in self.train_patient_info[train[0]]:
                                    self.train_patient_info[train[0]][resp] = 0
                                if resp not in self.test_patient_info[test[0]]:
                                    self.test_patient_info[test[0]][resp] = 0
                                self.train_patient_info[train[0]][resp] += 1
                                self.test_patient_info[test[0]][resp] += 1

                            elems = self.unique_second_element(train_both)
                            if len(elems) <= self.ordinal_threshold:
                                for e in elems:
                                    for train, test in zip(train_both, test_both):
                                        # Greater than one patientID per table
                                        # Categorical/ordinal response
                                        self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                        self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])
                                        resp = str(col[1])+'_'+str(e)
                                        if resp not in self.train_patient_info[train[0]]:
                                            self.train_patient_info[train[0]][resp] = 0
                                        if resp not in self.test_patient_info[test[0]]:
                                            self.test_patient_info[test[0]][resp] = 0

                                        self.train_patient_info[train[0]][resp] += int(train[1] == e)
                                        self.test_patient_info[test[0]][resp] += int(test[1] == e)
                            else:
                                for train, test in zip(train_both, test_both):
                                        # Multiple patientID
                                        # Multiple numerical responses
                                        # FIXME in postprocessing
                                        self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                        self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])

                                        resp = str(col[1])
                                        self.train_patient_info[train[0]][resp + '_'+str(self.
                                            train_patient_info[train[0]][str(resp +'_'+'count')])]= train[1]
                                        self.test_patient_info[test[0]][resp + '_'+str(self.
                                            test_patient_info[test[0]][str(resp +'_'+'count')])]= test[1]

                    elif col[1] in definitely_numeric:
                        pass
                    else:
                        pass





