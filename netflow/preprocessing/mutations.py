# import pandas as pd


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
