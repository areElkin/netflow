import json
import pandas as pd
from bravado.client import SwaggerClient

class CBioPortalClient:
    """ Swagger Client for querying study, patient, and sample-level metadata from cBioPortal.
    Parameters
    ----------
    url : `str`
        cBioPortal API specification URL (Currently: 'https://www.self.cbioportal.org/api/v3/api-docs').
    """
    def __init__(self, url='https://www.self.cbioportal.org/api/v3/api-docs'):
        self.cbioportal = SwaggerClient.from_url(url,
                                                 config={'validate_requests': False,
                                                         'validate_responses': False,
                                                         'validate_swagger_spec': False})


    def _read_response(self, response_wrapper):
        """ Convert Bravado response wrapper to a pandas DataFrame.
        Parameters
        ----------
        response_wrapper : `bravado.http_future.HttpFuture`
            Response returned by a Bravado API call.
        Returns
        -------
        result_df : `pandas.DataFrame`
            Flattened DataFrame constructed from the response body.
        """
        raw_response = response_wrapper.future.result()
        result_json = json.loads(raw_response.text)
        result_df = pd.json_normalize(result_json)
        return result_df
    
    
    # Study metadata
    def list_studies(self):
        """ Return names of all studies on cBioPortal.
        Returns
        -------
        study_names : `list` of `str`
            List of available study names from cBioPortal.
        """
    
        resp = self.cbioportal.Studies.getAllStudiesUsingGET()
        study_list = self._read_response(resp)
        study_names = study_list['name']
        return study_names
    
    
    def get_study_id(self, target_name, verbose=False):
        """ Return cBioPortal study ID given the study name.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.
        verbose : `bool`
            If `True` and a study matching `target_name` is not found, all valid study names
            are printed before throwing an error. (Default = False)
        Returns
        -------
        study_id : `str`
            cBioPortal-assigned study ID.
        """
        resp = self.cbioportal.Studies.getAllStudiesUsingGET()
        all_studies = self._read_response(resp)
    
        study_list = all_studies['name'].to_list()
        if target_name not in study_list:
            if verbose:
                print(f'Available studies: {study_list}')
            raise ValueError(f"No studies matching {target_name}")
        else:
            match_idx = study_list.index(target_name)
            study_id = all_studies.loc[match_idx,'studyId']
        return study_id
    
    
    def get_study_description(self, target_name):
        """ Get metadata for a cBioPortal study by name.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.
        Returns
        -------
        study_info : `pandas.DataFrame`
            Flat DataFrame containing all metadata fields for the study
            (e.g., name, description, cancer type, reference genome, sample count).
        """
        target_study_id = self.get_study_id(target_name)
        resp = self.cbioportal.Studies.getStudyUsingGET(studyId=target_study_id)
        study_info = self._read_response(resp)
        return study_info
    
    
    def list_attributes(self, target_name):
        """ List available clinical attributes (names) for a given study.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.
        Returns
        -------
        attr_list : `list` of `str`
            List of clinical attributes available in the study
            (e.g., ["OS_MONTHS", "OS_STATUS", "CANCER_TYPE", ...]).
        """
        target_study_id = self.get_study_id(target_name)
        resp = self.cbioportal.Clinical_Attributes.getAllClinicalAttributesInStudyUsingGET(studyId=target_study_id)
        clinical_attributes = self._read_response(resp)
        attr_list = clinical_attributes['clinicalAttributeId'].to_list()
        return attr_list
    
    
    def get_study_attribute(self, target_name, attribute):
        """ Get specified study-level attribute.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.
        attribute : `str`
            Name of the study-level attribute to retrieve.
            Options (to be extended)
            -------
            - 'data_types' : Returns a DataFrame of available molecular profile
              names and their corresponding cBioPortal profile IDs.
            - 'cancer_types' : Returns a DataFrame of cancer subtypes present in
              the study and the number of samples for each.
        Returns
        -------
        ans : `pandas.DataFrame`
            - If attribute='data_types': DataFrame with columns ['Name', 'ID'].
            - If attribute='cancer_types': DataFrame with columns ['Type', 'SampleCount'],
        """
        target_study_id = self.get_study_id(target_name)
        if attribute == 'data_types':
            resp = self.cbioportal.Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET(studyId=targetStudyId)
            molec_profile_info = self._read_response(resp)
            molec_profile_list = molec_profile_info['name'].to_list()
            molec_profile_ids = molec_profile_info.loc[molec_profile_info['name'].isin(molec_profile_list),'molecularProfileId']
            dtypes_dict = dict(zip(molec_profile_list, molec_profile_ids))
            molec_profile_df = pd.DataFrame(list(dtypes_dict.items()), columns=['Name', 'ID'])
            ans = molec_profile_df
        elif attribute == 'cancer_types':
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(
                            studyId=target_study_id,
                            attributeId="CANCER_TYPE")
            cancer_type_info = self._read_response(resp)
            cancer_type_df = cancer_type_info['value'].value_counts().reset_index()
            cancer_type_df.columns = ['Type', 'SampleCount']
            cancer_type_df = cancer_type_df.sort_values(by='Samples', ascending=False)
            ans = cancer_type_df.reset_index(drop=True)
        else:
            raise ValueError('Invalid input Attribute {attribute} requested.')
        return ans
    
    
    # Patient metadata
    def get_pt_attribute(self, target_study_id, attribute):
        """ Get requested patient-level attribute for all patients in a study.
        Parameters
        ----------
        target_study_id : `str`
            The cBioPortal study ID.
        attribute : `str`
            The patient-level attribute to retrieve.
            Options (to be expanded)
            -------
            - 'patient_id' : cBioPortal patient identifiers.
            - 'OSmonths' : Overall survival duration in months (float).
            - 'OSgroup' : Overall survival status (e.g., "LIVING" or "DECEASED").
        Returns
        -------
        attr : `list`
            Attribute values for all patients in study. Type varies by attribute (str or float):
        pts : `list` of `str`
            Corresponding patient IDs.
        """
        if attribute == 'patient_id':
            resp = self.cbioportal.Patients.getAllPatientsInStudyUsingGET(studyId=target_study_id)
            pt_info = self._read_response(resp)
            attr = pt_info['patientId'].to_list()
            pts = attr
        elif attribute == 'OSmonths':
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=target_study_id,
                                                                                attributeId='OS_MONTHS',
                                                                                clinicalDataType='PATIENT')
            pt_info = self._read_response(resp)
            attr = pt_info['value'].astype(float).to_list()
            pts = pt_info['patientId'].to_list()
        elif attribute == 'OSgroup':
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=target_study_id,
                                                                                attributeId='OS_STATUS',
                                                                                clinicalDataType='PATIENT')
            pt_info = self._read_response(resp)
            attr = pt_info['value'].to_list()
            pts = pt_info['patientId'].to_list()
        else:
            raise ValueError('Invalid input Attribute {attribute} requested.')
        return attr, pts
    
    
    def get_pt_info(self, target_name, attribute_list):
        """ Patient-level metadata for a given study .
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.
        attribute_list : `list` of `str`
            List of patient attributes to retrieve. Supported options are
            valid attribute arguments accepted by `get_pt_attribute`
        Returns
        -------
        pt_df : `pandas.DataFrame`
            DataFrame indexed by 'patientId' with columns holding requested attributes (NaN if missing).
        """
        target_study_id = self.get_study_id(target_name)
        pt_id_list, __ = self.get_pt_attribute(target_study_id, 'patient_id')
        pt_df = pd.DataFrame(pt_id_list, columns=['patientId'])
        for attr in attribute_list:
            vals, pts = self.get_pt_attribute(target_study_id, attr)
            temp = pd.DataFrame({'patientId': pts, attr: vals})
            pt_df = pt_df.merge(temp, on='patientId', how='left')
        return pt_df
    
    
    # Sample metadata
    def get_sample_attribute(self, target_study_id, attr):
        """ Get specified sample-level attribute for all samples in selected study.
        Parameters
        ----------
        target_study_id : `str`
            The cBioPortal study ID.
        attr : `str`
            The sample-level attribute to retrieve.
            Options
            -------
            'sample_id','patient_id','primary_site','sample_type', 'tumor_purity'
        Returns
        -------
        attr : `list`
            Attribute values, one per sample.
        samples : `list` of `str`
            Corresponding sample IDs.
        """
        if attr == 'sample_id':
            resp = self.cbioportal.Samples.getAllSamplesInStudyUsingGET(studyId=target_study_id)
            sample_info = self._read_response(resp)
            attr = sample_info['sampleId'].to_list()
            samples = attr
        elif attr == 'patient_id':
            resp = self.cbioportal.Samples.getAllSamplesInStudyUsingGET(studyId=target_study_id)
            sample_info = self._read_response(resp)
            attr = sample_info['patientId'].to_list()
            samples = sample_info['sampleId'].to_list()
        elif attr == 'primary_site':
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=target_study_id,
                                                                              attributeId='PRIMARY_SITE')
            sample_info = self._read_response(resp)
            attr = sample_info['value'].to_list()
            samples = sample_info['sampleId'].to_list()
        elif attr == 'sample_type':
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=target_study_id,
                                                                              attributeId='SAMPLE_TYPE')
            sample_info = self._read_response(resp)
            attr = sample_info['value'].to_list()
            samples = sample_info['sampleId'].to_list()
        elif attr == 'tumor_purity':
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=target_study_id,
                                                                              attributeId='TUMOR_PURITY')
            sample_info = self._read_response(resp)
            attr = sample_info['value'].to_list()
            samples = sample_info['sampleId'].to_list()
        else:
            raise ValueError (f'No attribute {attr}.')
        return attr, samples
    
    
    def get_sample_info(self, target_name, attr_list):
        """ Get sample-level metadata for specified study.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.
        attr_list : `list` of `str`
            List of sample attributes to retrieve. Options are
            valid attr argument accepted by `get_sample_attribute`
        Returns
        -------
        sample_df : `pandas.DataFrame`
            DataFrame indexed by 'sampleId' with columns holding requested attributes (NaN if missing).
        """
        target_study_id = self.get_study_id(target_name)
        sample_id_list, __ = self.get_sample_attribute(target_study_id, 'sample_id')
        sample_df = pd.DataFrame(sample_id_list, columns=['sampleId'])
        if 'patient_id' not in attr_list:
            attr_list.insert(0, 'patient_id')
        for attr in attr_list:
            vals, samples = self.get_sample_attribute(target_study_id, attr)
            temp = pd.DataFrame({'sampleId': samples, attr: vals})
            sample_df = sample_df.merge(temp, on='sampleId', how='left')
        return sample_df
    
    
    def map_samples_to_pts(self, sample_df, pt_df):
        """ Associate sample-level metadata with a patient ids.
        Parameters
        ----------
        sample_df : `pandas.DataFrame`
            Sample-level metadata as returned by `get_sample_info`.
        pt_df : `pandas.DataFrame`
            Patient-level metadata as returned by :func:`get_pt_info`. Must include 'patientId' column.
    
        Returns
        -------
        summary_df : `pandas.DataFrame`
            Merged DataFrame with sample rows associated with patient ids.
        """
        summary_df = pd.merge(sample_df, pt_df, on='patientId', how='left')
        #summary_df = summary_df.set_index('sampleId')
        return summary_df