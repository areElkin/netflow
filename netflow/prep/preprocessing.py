import numpy as np
import pandas as pd
import sklearn.decomposition


def MAF_to_binary_matrix(mut, vc_col='Variant_Classification',
                         gene_col='Hugo_Symbol', sample_col='Tumor_Sample_Barcode',
                         vcs=None, samples=None):
    """ Convert mutations from MAF style mutation file to binary mutation matrix.

    Parameters
    ----------
    mut : `pandas.DataFrame`
        Mutation Annotation Format (MAF)-style data (primarily from TCGA/GDC, designed for
        storing aggregated somatic mutation data, containing one mutation per row).
    vc_col : `str`
        Header of column containing variant classification.
        Typically "Variant_Classification" or "VARIANT_CLASSIFICATION". (Default = ""Variant_Classification")
    gene_col : `str`
        Header of column containing the gene name.
        Typically "Hugo_Symbol", "HUGO_SYMBOL", "Entrez_Gene_Id", "ENTREZ_GENE_ID". (Default = "Hugo_Symbol")
    sample_col : `str`
        Header of column containing the sample name.
        Typically "Tumor_Sample_Barcode", "SAMPLE_ID", or "Tumor_Sample". (Default="Tumor_Sample_Barcode")
    vcs : {`None`, 'all', list-like}
        (Optional) Specify variant classes to be considered.

        Options
        -------
        -`None` : (Default) Considers nonsynonymous mutations with the following classifications:
            ``vcs={"Missense_Mutation","Nonsense_Mutation","Frame_Shift_Del","Frame_Shift_Ins",
                   "In_Frame_Del","In_Frame_Ins","Splice_Site","Translation_Start_Site","Nonstop_Mutation"}``.
        - `'all'` : All mutations are included.
        - list-like : Restrict to mutations with ``vc_col`` in the given list.
    samples : list-like
        (Optional) If provided, restrict to list of provided samples.

    Returns
    -------
    M : `pandas.DataFrame`
        The binary matrix indicating nonsynonymous mutations (sample x gene).
    """
    mut2 = mut.copy()

    # only keep selected samples
    if samples is not None:
        mut2 = mut2[mut2[sample_col].astype(str).isin(samples)].copy()

        # filter mutations
    if vcs is None:
        vcs = {
            "Missense_Mutation", "Nonsense_Mutation", "Frame_Shift_Del", "Frame_Shift_Ins",
            "In_Frame_Del", "In_Frame_Ins", "Splice_Site", "Translation_Start_Site", "Nonstop_Mutation",
        }
    if vcs != 'all':
        mut2 = mut2[mut2[vc_col].isin(vcs)]  # only keep non-synonymous mutations

    # make mutation matrix
    M = mut2.groupby([sample_col, gene_col]).size().unstack().fillna(0)

    # make binary
    M = (M > 0).astype(int)
    return M

def PCA_tx(data, n_components=None, random_state=None):
        """ Principal component analysis (PCA) decomposition.

        Parameters
        ----------
        data : {`np.array`}
            Data (num_observations, num_features) on which PCA decomposition will be performed.
        n_components : {`None`, `int`}
            The number of principal components to keep.
            If `None`, all principal components are kept.
        random_state : {`None`, `int`}
            Random state used for certain solvers. Pass an `int` for reproducible results across runs.

        Returns
        -------
        PCA data is returned.
        """
        pca_obj = sklearn.decomposition.PCA(n_components=n_components, random_state=random_state)
        data_pca = pca_obj.fit_transform(data)

        return data_pca

def log1p_tx(data, base=None):
        """ Logarithmic data transformation.

        Computes :math:`data = \\log(data + 1)` with the natural logarithm as the default base.

        Parameters
        ----------
        data : {`np.array`}
            Data (num_features, num_observations) to be logarithmically transformed.
        base : {`None`, `int`}
            Base used for the logarithmic transformation.

        Returns
        -------
        Logarithmically transformed data (num_features, num_observations).
        """
        if not (np.issubdtype(data.dtype, np.floating) or np.issubdtype(data.dtype, complex)):
            data = data.astype(float)
        data_log = np.log1p(data)
        if base is not None:
            np.divide(data_log, np.log(base), out=data_log)

        return data_log


def rand_offset_tx(data, scale, center=0., rand_seed=None):
        """ Add random noise to the data.

        Parameters
        ----------
        data : `np.array`
            Data (num_features, num_observations) to be logarithmically transformed.
        scale : `float` (non-negative)
             Standard deviation of the Gaussian distribution used to add random noise.
        center: `float`
             Mean of the Gaussian distribution used to add random noise.
             If unspecified, the default center is 0.
        rand_seed : {`None`, `int`}
            Seed (`int`) for reproducible results.


        Returns
        -------
        data_unique: `np.array`
         Data with random noise added to ensure unique columns(num_features, num_observations).

        """
        if rand_seed is not None:
            rng = np.random.default_rng(seed=rand_seed)
            noise = rng.normal(center, scale, data.shape)
        else:
            noise = np.random.normal(center, scale, data.shape)

        data_unique = data + noise
        return data_unique

