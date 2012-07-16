#! /usr/bin/env python
"""
Diabetus
"""
from os import path
import sqlite3
import re
from collections import Counter
import numpy as np
import pandas

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
        self.uniq_threshold = 0.001
        self.datamart = pandas.HDFStore(path.join(self.data_dir, 'data.h5'))

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

    def most_unique(self, list_o_stuff):
        c = Counter(list_o_stuff)
        size = len(list_o_stuff)
        common_stuff = []
        for stuff, count in  c.most_common():
            if float(count)/float(size) <= self.uniq_threshold:
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
        super_query = self.conn.execute("SELECT PatientGuid, {0} FROM {1}".format(
            column_name, table))
        return [s for s in super_query.fetchall() if s[1] != None]

    def get_Y_train(self):
        diabetes_cursor = self.conn.execute("SELECT PatientGuid,\
                dmindicator FROM training_patient")
        diabetes = diabetes_cursor.fetchall()
        return diabetes

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
                    train_both = self.get_tables_column_patient(
                            train_table, col[1])
                    test_both = self.get_tables_column_patient(
                            test_table, col[1])
                    elems = self.unique_second_element(train_both)
                    resp = str(table.format('wut') + col[1])

                    if self.test_if_fake_int(train_values):
                        if self.is_list_unique(train_both):
                            if len(elems) <= self.ordinal_threshold:
                                print("Type A")
                                print(table, col)
                                for e in elems:
                                    # Ignore second element of binary variables
                                    if len(elems) == 2 and elems[1] == e:
                                        pass
                                    else:
                                       for train, test in zip(train_both, test_both):
                                            # One patientID per table
                                            # Categorical/ordinal responses
                                            self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                            self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])

                                            self.train_patient_info[train[0]][resp+'_'+str(e)] = int(train[1] == e)
                                            self.test_patient_info[test[0]][resp+'_'+str(e)] = int(test[1] == e)
                            else:
                                print("Type B")
                                print(table, col)
                                for train, test in zip(train_both, test_both):
                                    # One patientID per table
                                    # Lots of numerical responses
                                    self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                    self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])

                                    self.train_patient_info[train[0]][resp] = train[1]
                                    self.test_patient_info[test[0]][resp] = test[1]
                        else:
                            for train, test in zip(train_both, test_both):
                            # Count table occurences
                                self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])
 
                                resp_cn = resp + str('_'+'count')
                                if resp_cn not in self.train_patient_info[train[0]]:
                                    self.train_patient_info[train[0]][resp_cn] = 0
                                if resp_cn not in self.test_patient_info[test[0]]:
                                    self.test_patient_info[test[0]][resp_cn] = 0
                                self.train_patient_info[train[0]][resp_cn] += 1
                                self.test_patient_info[test[0]][resp_cn] += 1

                            if len(elems) <= self.ordinal_threshold:
                                print("Type C")
                                print(table, col)
                                for e in elems:
                                    for train, test in zip(train_both, test_both):
                                        # Greater than one patientID per table
                                        # Categorical/ordinal responses
                                        self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                        self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])
                                        resp_e = resp + '_'+str(e)
                                        if resp_e not in self.train_patient_info[train[0]]:
                                            self.train_patient_info[train[0]][resp_e] = 0
                                        if resp_e not in self.test_patient_info[test[0]]:
                                            self.test_patient_info[test[0]][resp_e] = 0
                                        self.train_patient_info[train[0]][resp_e] += int(train[1] == e)
                                        self.test_patient_info[test[0]][resp_e] += int(test[1] == e)
                            else:
                                cn_ids = {}
                                cn_ids['train'] = {}
                                cn_ids['test'] = {}
                                print("Type D")
                                print(table, col)
 
                                for train, test in zip(train_both, test_both):
                                    # Multiple patientID
                                    # Multiple numerical responses
                                    # FIXME in postprocessing
                                    self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                    self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])
                                    if train[0] not in cn_ids['train']:
                                        cn_ids['train'][train[0]] = 0
                                    else:
                                        cn_ids['train'][train[0]] += 1
                                    if test[0] not in cn_ids['test']:
                                        cn_ids['test'][test[0]] = 0
                                    else:
                                        cn_ids['test'][test[0]] += 1
                                    resp = str(col[1])
                                    self.train_patient_info[train[0]][resp + '_DUMMY_'+str(
                                        cn_ids['train'][train[0]])]= train[1]
                                    self.test_patient_info[test[0]][resp + '_DUMMY_'+str(
                                        cn_ids['test'][test[0]])]= test[1]

                    elif col[1] in definitely_numeric:
                        # FIXME?: ignores dosage quantities.
                        pass
                    else:
                        # Categorical/Nominal
                        most_elements = self.most_unique([t[1] for t in train_both])
                        print("Type E")
                        print(table, col)
                        for e in most_elements:
                            for train, test in zip(train_both, test_both):
                                self.train_patient_info = self.add_subdict(self.train_patient_info, train[0])
                                self.test_patient_info = self.add_subdict(self.test_patient_info, test[0])
                                resp_e = resp+'_'+str(e)

                                if resp_e not in self.train_patient_info[train[0]]:
                                    self.train_patient_info[train[0]][resp_e] = 0
                                if resp_e not in self.test_patient_info[test[0]]:
                                    self.test_patient_info[test[0]][resp_e] = 0

                                self.train_patient_info[train[0]][resp_e] += int(train[1] == e)
                                self.test_patient_info[test[0]][resp_e] += int(test[1] == e)

    def run(self):
        self.get_datasets()

        Y_train = self.get_Y_train()
        for patientID, dmindicator in Y_train:
            if patientID not in self.train_patient_info:
                print("{0} ignored".format(patientID))
                pass
            else:
                self.train_patient_info[patientID]['Target'] = dmindicator

        self.train_df = pandas.DataFrame(self.train_patient_info)
        del self.train_patient_info
        self.test_df = pandas.DataFrame(self.test_patient_info)
        del self.test_patient_info

        del self.train_df['patientGuid']
        indices = self.train_df.index.tolist()
        self.train_df = self.train_df.transpose()
        for i in indices:
            self.train_df[i][self.train_df[i] == 'NULL'] = np.nan
        self.train_df = self.train_df.astype(float)

        easy_rm = []
        for i in indices:
            if re.search('PatientGuid', i):
                easy_rm.append(i)
        for i in easy_rm:
            del self.train_df[i]

        easy_rm = []
        for i in indices:
            if re.search('DUMMY', i):
                easy_rm.append(i)

        index_info = set([])
        for i in easy_rm:
            lol = i.split('_')[0]
            if lol not in index_info:
                index_info.add(lol)
        index_dict = {}
        for i in index_info:
            index_dict[i] = []
            for j in easy_rm:
                if re.match(i, j.split('_')[0]):
                    index_dict[i].append(j)

        for name,sub in index_dict.items():
            self.train_df[name+'_max'] = self.train_df[sub].max(axis=1, skipna=True)
            self.train_df[name+'_min'] = self.train_df[sub].min(axis=1, skipna=True)
            self.train_df[name+'_median'] = self.train_df[sub].median(axis=1, skipna=True)
            self.train_df[name+'_std'] = self.train_df[sub].std(axis=1, skipna=True)
            for i in sub:
                del self.train_df[i]

        del self.test_df['patientGuid']
        indices = self.test_df.index.tolist()
        self.test_df = self.test_df.transpose()
        for i in indices:
            self.test_df[i][self.test_df[i] == 'NULL'] = np.nan
        self.test_df = self.test_df.astype(float)

        easy_rm = []
        for i in indices:
            if re.search('PatientGuid', i):
                easy_rm.append(i)
        for i in easy_rm:
            del self.test_df[i]

        easy_rm = []
        for i in indices:
            if re.search('DUMMY', i):
                easy_rm.append(i)

        index_info = set([])
        for i in easy_rm:
            lol = i.split('_')[0]
            if lol not in index_info:
                index_info.add(lol)
        index_dict = {}
        for i in index_info:
            index_dict[i] = []
            for j in easy_rm:
                if re.match(i, j.split('_')[0]):
                    index_dict[i].append(j)

        for name,sub in index_dict.items():
            self.test_df[name+'_max'] = self.test_df[sub].max(axis=1, skipna=True)
            self.test_df[name+'_min'] = self.test_df[sub].min(axis=1, skipna=True)
            self.test_df[name+'_median'] = self.test_df[sub].median(axis=1, skipna=True)
            self.test_df[name+'_std'] = self.test_df[sub].std(axis=1, skipna=True)
            for i in sub:
                del self.test_df[i]
        self.datamart['train'] = self.train_df
        self.datamart['test'] = self.test_df

    def get_munged_clean_data(self):
        train_df = self.datamart['train']

        Y_train_df = train_df['Target']
        Y_train = np.array(Y_train_df)
        del train_df['Target']

        test_df = self.datamart['test']

        assert np.all(train_df.columns == test_df.columns)
        X_train = np.array(train_df)
        X_test = np.array(test_df)
        X_train_nan = np.isnan(X_train)
        X_test_nan = np.isnan(X_test)
        X_train = np.hstack((X_train,X_train_nan))
        X_test = np.hstack((X_test,X_test_nan))
        X_train[np.isnan(X_train)] = 0.0
        X_test[np.isnan(X_test)] = 0.0
        return X_train, Y_train, X_test

