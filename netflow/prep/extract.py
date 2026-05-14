import json
import pandas as pd
import requests
from bravado.client import SwaggerClient

class CBioPortalClient:
    """ Swagger Client for querying study, patient, and sample-level metadata from cBioPortal.
    Parameters
    ----------
    url : `str`
        cBioPortal API specification URL (Currently: 'https://www.self.cbioportal.org/api/v3/api-docs').
    """
    BODY_PARAM_PATCHES = {
    '/api/molecular-profiles/{molecularProfileId}/mutations/fetch': {
        'post': {
            'name': 'mutationFilter',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'sampleListId':  {'type': 'string'},
                    'sampleIds':     {'type': 'array', 'items': {'type': 'string'}},
                    'entrezGeneIds': {'type': 'array', 'items': {'type': 'integer'}},
                }
            }
        }
    },
    '/api/molecular-profiles/{molecularProfileId}/molecular-data/fetch': {
        'post': {
            'name': 'molecularDataFilter',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'sampleListId':  {'type': 'string'},
                    'sampleIds':     {'type': 'array', 'items': {'type': 'string'}},
                    'entrezGeneIds': {'type': 'array', 'items': {'type': 'integer'}},
                }
            }
        }
    },
    '/api/molecular-profiles/{molecularProfileId}/discrete-copy-number/fetch': {
        'post': {
            'name': 'discreteCopyNumberFilter',
            'in': 'body',
            'required': True,
            'schema': {
                'type': 'object',
                'properties': {
                    'sampleListId': {'type': 'string'},
                    'sampleIds': {'type': 'array', 'items': {'type': 'string'}},
                    'entrezGeneIds': {'type': 'array', 'items': {'type': 'integer'}},
                }
            }
        }
    },
    }

    SAMPLE_ATTR_MAP = {
        'patient_id': 'patientId',
        'sample_collection_source': 'sampleCollectionSource',
        'specimen_preservation_type': 'specimenPreservationType',
        'specimen_type': 'specimenType',
        'dna_input': 'dnaInput',
        'sample_coverage': 'sampleCoverage',
        'tumor_purity': 'tumorPurity',
        'matched_status': 'matchedStatus',
        'sample_type': 'sampleType',
        'primary_site': 'primaryTumorSite',
        'metastatic_site': 'metastaticSite',
        'sample_class': 'sampleClass',
        'oncotree_code': 'oncotreeCode',
        'cancer_type': 'cancerType',
        'cancer_type_detailed': 'cancerTypeDetailed',
        'somatic_status': 'somaticStatus',
        'tmb_nonsynonymous': 'tmbNonsynonymous',
        'sequenced': 'sequenced',
        'copy_number': 'copyNumber',
        'mrna_data': 'mrnaData',
        'msi_status': 'msiStatus',
        'methylation_status': 'methylationStatus',
        'methylation_subtype': 'methylationSubtype',
        'icluster': 'icluster',
        'mlh1_silencing': 'mlh1Silencing',
        'gene_expression_subtype': 'geneExpressionSubtype',
        'hyper_mutated': 'hyperMutated'
    }

    PT_ATTR_MAP = {
        'sex': 'sex',
        'smoking_history': 'smokingHistory',
        'os_months': 'osMonths',
        'os_status': 'osStatus',
        'tumor_histologic_subtype': 'tumorHistologicSubtype',
        'primary_tumor_pathologic_spread': 'primaryTumorPathologicSpread'
    }

    def __init__(self, url='https://www.cbioportal.org/api/v3/api-docs'):
        raw_spec = requests.get(url).json()
        patched_spec = self._patch_spec(raw_spec)

        self.cbioportal = SwaggerClient.from_spec(patched_spec,
                                                  origin_url=url,
                                                  config={'validate_requests': False,
                                                         'validate_responses': False,
                                                         'validate_swagger_spec': False})

    def _patch_spec(self, spec):
        """Patch to pass requestBody end-points as Swagger req. body params (passed through Bravado)"""
        for path, methods in self.BODY_PARAM_PATCHES.items():
            for method, body_param in methods.items():
                if path in spec.get('paths', {}):
                    params = spec['paths'][path][method].setdefault('parameters', [])
                    spec['paths'][path][method].pop('requestBody', None)
                    params.append(body_param)
        return spec


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


    # Study metadata
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
    
    
    def list_study_attributes(self, target_name):
        """ List clinical attributes available in a given study.
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
    

    def list_study_data(self, target_name):
        """ List molecular data types available in a given study.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.
        Returns
        -------
        data_list : `list` of `str`
            List of molecular data types (names) available in the study (e.g., ["COPY_NUMBER_ALTERATION", "MRNA_EXPRESSION", ...]).
        id_list : `list` of `str`
            List of molecular data IDs is returned.
        """
        target_study_id = self.get_study_id(target_name)
        resp = self.cbioportal.Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET(studyId=target_study_id)
        mp_info = self._read_response(resp)

        data_list = mp_info['molecularAlterationType'].to_list()
        id_list = mp_info['molecularProfileId'].to_list()
        return data_list, id_list


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
            resp = self.cbioportal.Molecular_Profiles.getAllMolecularProfilesInStudyUsingGET(studyId=target_study_id)
            molec_profile_info = self._read_response(resp)
            molec_profile_list = molec_profile_info['name'].to_list()
            molec_profile_ids = molec_profile_info.loc[molec_profile_info['name'].isin(molec_profile_list),'molecularProfileId']
            dtypes_dict = dict(zip(molec_profile_list, molec_profile_ids))
            molec_profile_df = pd.DataFrame(list(dtypes_dict.items()), columns=['name', 'id'])
            ans = molec_profile_df
        elif attribute == 'cancer_types':
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(
                            studyId=target_study_id,
                            attributeId="CANCER_TYPE")
            cancer_type_info = self._read_response(resp)
            cancer_type_df = cancer_type_info['value'].value_counts().reset_index()
            cancer_type_df.columns = ['type', 'sampleCount']
            cancer_type_df = cancer_type_df.sort_values(by='sampleCount', ascending=False)
            ans = cancer_type_df.reset_index(drop=True)
        else:
            raise ValueError(f'Invalid input Attribute {attribute} requested.')
        return ans
    
    
    # Patient metadata
    def get_pt_attribute(self, target_study_id, attr):
        """ Get requested patient-level attribute for all patients in a study.
        Parameters
        ----------
        target_study_id : `str`
            The cBioPortal study ID.
        attribute : `str`
            The patient-level attribute to retrieve.
            For valid options, see PT_ATTR_MAP.

        Returns
        -------
        attr : `list`
            Attribute values for all patients in study. Type varies by attribute (str or float).
        pts : `list` of `str`
            Corresponding patient IDs.
        """
        valid_attrs = list(self.PT_ATTR_MAP.keys())

        if attr == 'patient_id':
            resp = self.cbioportal.Patients.getAllPatientsInStudyUsingGET(studyId=target_study_id)
            pt_info = self._read_response(resp)
            vals = pt_info['patientId'].to_list()
            pts = vals
        else:
            if attr not in valid_attrs:
                raise ValueError(f"No patient attribute '{attr}'. Valid options:\n  {valid_attrs}")

            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(studyId=target_study_id,
                                                                                attributeId=attr.upper(),
                                                                                clinicalDataType='PATIENT')
            pt_info = self._read_response(resp)
            vals = pt_info['value']
            if attr == 'os_months':
                vals = vals.astype(float)
            vals = vals.to_list()
            pts = pt_info['patientId'].to_list()

        return vals, pts
    
    
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
        attr_dict = self.PT_ATTR_MAP
        target_study_id = self.get_study_id(target_name)
        pt_id_list, __ = self.get_pt_attribute(target_study_id, 'patient_id')
        pt_df = pd.DataFrame(pt_id_list, columns=['patientId'])
        for attr in attribute_list:
            try:
                vals, pts = self.get_pt_attribute(target_study_id, attr)
                temp = pd.DataFrame({'patientId': pts, attr_dict[attr]: vals})
            except Exception:
                temp = pd.DataFrame({'patientId': pt_id_list, attr_dict[attr]: pd.NA})
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
            For valid options, see SAMPLE_ATTR_MAP.
        Returns
        -------
        attr : `list`
            Attribute values, one per sample.
        samples : `list` of `str`
            Corresponding sample IDs.
        """

        valid_attrs = list(self.SAMPLE_ATTR_MAP.keys())
        if attr == 'sample_id':
            resp = self.cbioportal.Samples.getAllSamplesInStudyUsingGET(studyId=target_study_id)
            sample_info = self._read_response(resp)
            vals = sample_info['sampleId'].to_list()
            samples = vals
        elif attr == 'patient_id':
            resp = self.cbioportal.Samples.getAllSamplesInStudyUsingGET(studyId=target_study_id)
            sample_info = self._read_response(resp)
            vals = sample_info['patientId'].to_list()
            samples = sample_info['sampleId'].to_list()
        else:
            if attr not in valid_attrs:
                raise ValueError(f"No attribute '{attr}'. Valid options: {sorted(valid_attrs)}")
            resp = self.cbioportal.Clinical_Data.getAllClinicalDataInStudyUsingGET(
                studyId=target_study_id, attributeId=attr.upper())
            sample_info = self._read_response(resp)
            vals = sample_info['value'].to_list()
            samples = sample_info['sampleId'].to_list()
        return vals, samples
    
    
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
            DataFrame indexed by 'sample_id' with columns holding requested attributes (NaN if missing).
        """
        attr_dict = self.SAMPLE_ATTR_MAP
        target_study_id = self.get_study_id(target_name)
        sample_id_list, __ = self.get_sample_attribute(target_study_id, 'sample_id')

        sample_df = pd.DataFrame(sample_id_list, columns=['sampleId'])
        if 'patient_id' not in attr_list:
            attr_list = ['patient_id'] + attr_list
        for attr in attr_list:
            try:
                vals, samples = self.get_sample_attribute(target_study_id, attr)
                temp = pd.DataFrame({'sampleId':samples, attr_dict[attr]:vals})
            except Exception:
                temp = pd.DataFrame({'sampleId':sample_id_list, attr_dict[attr]:pd.NA})
            sample_df = sample_df.merge(temp, on='sampleId', how='left')
            sample_df = sample_df.rename(columns={'patient_id': 'patientId'})
        return sample_df


    @staticmethod
    def map_samples_to_pts(sample_df, pt_df):
        """ Associate sample-level metadata with a patient ids.
        Parameters
        ----------
        sample_df : `pandas.DataFrame`
            Sample-level metadata as returned by `get_sample_info`.
        pt_df : `pandas.DataFrame`
            Patient-level metadata as returned by :func:`get_pt_info`. Must include 'patient_id' column.
    
        Returns
        -------
        summary_df : `pandas.DataFrame`
            Merged DataFrame with sample rows associated with patient ids.
        """
        summary_df = pd.merge(sample_df, pt_df, on='patientId', how='left')
        #summary_df = summary_df.set_index('sample_id')
        return summary_df


    def map_entrezid_to_hugosymbol(self, input_df):
        """ Map gene IDs to HUGO nomenclature.
        Parameters
        ----------
        input_df : `pandas.DataFrame`
            Input dataframe containing `entrezGeneId` column.

        Returns
        -------
        input_df : `pandas.DataFrame`
            DataFrame with column `hugoGeneSymbol` corresponding to entrezGeneIds.
        """

        input_df = input_df.copy()
        resp = self.cbioportal.Genes.getAllGenesUsingGET()
        gene_info = self._read_response(resp)
        gene_map = gene_info.set_index('entrezGeneId')['hugoGeneSymbol'].to_dict()
        input_df['hugoGeneSymbol'] = input_df['entrezGeneId'].map(gene_map).fillna("Unknown")
        return input_df


    def get_clinical_data(self, target_name):
        """ Extract patient clincal data for a given study.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.

        Returns
        -------
        clin_df : `pandas.DataFrame`
            DataFrame containing available clinical data for all patients in the study.
        """
        sample_df = self.get_sample_info(target_name, ['primary_site', 'sample_type', 'tumor_purity'])
        pt_df = self.get_pt_info(target_name, ['os_months', 'os_status'])
        clin_df = CBioPortalClient.map_samples_to_pts(sample_df, pt_df)
        clin_df = clin_df.set_index('sampleId')
        clin_df['osGroup'] = clin_df['osStatus'].str.contains('DECEASED', na=False).astype(float)
        return clin_df


    def get_cna_data(self, target_name):
        """ Extract copy number alterations in a given study.
        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.

        Returns
        -------
        cna_raw : `pandas.DataFrame`
            DataFrame containing available CNA data for all samples in the study.

        cna_df : `pandas.DataFrame`
            cna_raw re-organized for loading into the keeper.
        """

        target_study_id = self.get_study_id(target_name)

        # Check if available
        sel_type = 'COPY_NUMBER_ALTERATION'
        cna_types, cna_ids = self.list_study_data(target_name)
        if sel_type not in cna_types:
            raise ValueError(f"No profile matching {sel_type} found in study {target_name}.")
        else:
            match_idx = cna_types.index(sel_type)
            mpid = cna_ids[match_idx] #TBD: Check if multiple IDs can have the same study data type

        # Get CNA data
        resp = self.cbioportal.Discrete_Copy_Number_Alterations.fetchDiscreteCopyNumbersInMolecularProfileUsingPOST(
                molecularProfileId=mpid,
                discreteCopyNumberFilter={'sampleListId': target_study_id + '_all'},
                discreteCopyNumberEventType='ALL',
                projection='SUMMARY')
        cna_raw = self._read_response(resp)

        # Map entrezGeneIds to hugoGeneSymbols
        cna_raw = self.map_entrezid_to_hugosymbol(cna_raw)

        # Reshape to (n_feats, n_obs)
        cna_df = cna_raw.pivot(index='hugoGeneSymbol', columns='sampleId', values='alteration')
        cna_df = cna_df.apply(pd.to_numeric, errors='coerce')
        cna_df.index.name = 'hugoGeneSymbol'
        cna_df.columns.name = None
        cna_df = cna_df.fillna(0).astype(int)
        return cna_raw, cna_df


    def get_mutation_data(self, target_name):
        """ Get mutation data for all samples in a study from the MUTATION_EXTENDED molecular profile.

        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.

        Returns
        -------
        mut_raw : `pandas.DataFrame`
            DataFrame containing available mutation data for all samples in the study.
        mut_df : `pandas.DataFrame`
            mut_raw re-organized for loading into the keeper.
        """

        target_study_id = self.get_study_id(target_name)

        # Check if available
        sel_type = 'MUTATION_EXTENDED'
        mut_types, mut_ids = self.list_study_data(target_name)
        if sel_type not in mut_types:
            raise ValueError(f"No profile matching {sel_type} found in study {target_name}.")
        else:
            match_idx = mut_types.index(sel_type)
            mpid = mut_ids[match_idx]

        # Fetch mutations
        resp = self.cbioportal.Mutations.fetchMutationsInMolecularProfileUsingPOST(
                molecularProfileId=mpid,
                mutationFilter={'sampleListId': target_study_id + '_all'},
                projection='SUMMARY')
        mut_raw = self._read_response(resp)
        mut_raw = self.map_entrezid_to_hugosymbol(mut_raw)

        # Reshape to (n_feats, n_obs)
        mut_df = (mut_raw.groupby(['hugoGeneSymbol', 'sampleId'])
            .size()
            .unstack(fill_value=0)
            .clip(upper=1))
        mut_df.index.name = 'hugoGeneSymbol'
        mut_df.columns.name = None
        return mut_raw, mut_df


    def get_methylation_data(self, target_name):
        """ Get DNA methylation (beta values) for all samples in a study from the METHYLATION molecular profile.

        Parameters
        ----------
        target_name : `str`
            Name of a study exactly as displayed on cBioPortal.

        Returns
        -------
        methyl_raw : `pandas.DataFrame`
            Beta-value dataframe (n_genes, n_samples).
            Values are floats in [0, 1] where 0 = unmethylated, 1 = fully methylated). NaNs indicate missing data.
        methyl_df : `pandas.DataFrame`
            methyl_raw processed for loading into the keeper.
        """

        target_study_id = self.get_study_id(target_name)

        # Check if available
        sel_type = 'METHYLATION'
        methyl_types, methyl_ids = self.list_study_data(target_name)
        if sel_type not in methyl_types:
            raise ValueError(f"No profile matching {sel_type} found in study {target_name}.")
        else:
            match_idx = methyl_types.index(sel_type)
            mpid = methyl_ids[match_idx]

        # Fetch methylation values
        resp = self.cbioportal.Molecular_Data.fetchAllMolecularDataInMolecularProfileUsingPOST(
            molecularProfileId=mpid,
            molecularDataFilter={'sampleListId': target_study_id + '_all'},
            projection='SUMMARY')
        methyl_raw = self._read_response(resp)
        methyl_raw = self.map_entrezid_to_hugosymbol(methyl_raw)

        methyl_df = methyl_raw.copy()
        methyl_df['value'] = pd.to_numeric(methyl_raw['value'], errors='coerce')

        # Aggregate methylation by mean & reshape to (n_feats, n_obs)
        methyl_df = (
            methyl_df.groupby(['hugoGeneSymbol', 'sampleId'])['value']
            .mean()
            .unstack(level='sampleId')
        )
        methyl_df.index.name = 'hugoGeneSymbol'
        methyl_df.columns.name = None
        return methyl_raw, methyl_df


def get_rna_seq_data(self, target_name):
    """Get RNA-seq expression data for all samples in a study.

    Parameters
    ----------
    target_name : `str`
        Name of a study exactly as displayed on cBioPortal.

    Returns
    -------
    rna_raw : `pandas.DataFrame`
        DataFrame containing RNA-seq expression values for all samples in the study.
    rna_df : `pandas.DataFrame`
        rna_raw processed for loading into the keeper.

    """
    target_study_id = self.get_study_id(target_name)

    sel_type = 'MRNA_EXPRESSION'
    rna_types, rna_ids = self.list_study_data(target_name)
    if sel_type not in rna_types:
        raise ValueError(f"No profile matching {sel_type} found in study {target_name}.")
    else:
        match_idx = rna_types.index(sel_type)
        mpid = rna_ids[match_idx] #TBD: Check if multiple IDs can have the same study data type

    resp = self.cbioportal.Molecular_Data.fetchAllMolecularDataInMolecularProfileUsingPOST(
        molecularProfileId=mpid,
        molecularDataFilter={'sampleListId': target_study_id + '_all'},
        projection='SUMMARY')
    rna_raw = self._read_response(resp)
    rna_raw = self.map_entrezid_to_hugosymbol(rna_raw)

    rna_df = rna_raw.copy()
    rna_df['value'] = pd.to_numeric(rna_df['value'], errors='coerce')

    # Aggregate by mean, reshape to (n_feats, n_obs)
    rna_df = (
        rna_df.groupby(['hugoGeneSymbol', 'sampleId'])['value']
        .mean()
        .unstack(level='sampleId')
    )
    rna_df.index.name = 'hugoGeneSymbol'
    rna_df.columns.name = None

    return rna_raw, rna_df


def get_structural_variant_data(self, target_name):
    """Get structural variant (fusion) data for all samples in a study.

    Parameters
    ----------
    target_name : `str`
        Name of a study exactly as displayed on cBioPortal.

    Returns
    -------
    sv_df : `pandas.DataFrame`
        DataFrame containing structural variant data for all samples in the study.
    """
    target_study_id = self.get_study_id(target_name)

    sel_type = 'STRUCTURAL_VARIANT'
    sv_types, sv_ids = self.list_study_data(target_name)
    if sel_type not in sv_types:
        raise ValueError(f"No profile matching {sel_type} found in study {target_name}.")
    else:
        match_idx = sv_types.index(sel_type)
        mpid = sv_ids[match_idx]

    resp = self.cbioportal.Structural_Variants.fetchStructuralVariantsUsingPOST(
        molecularProfileId=mpid,
        structuralVariantFilter={'sampleListId': target_study_id + '_all'})
    sv_df = self._read_response(resp)
    sv_df = self.map_entrezid_to_hugosymbol(sv_df)
    return sv_df

