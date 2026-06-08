
def pretify_method_name(method_name):
    if method_name == 'DRVI-IK':
        return 'DRVI-AP'
    if method_name == 'DRVI-CVAE':
        return 'CVAE'
    if method_name == 'DRVI-sum-sp':
        return 'DRVI-APnoEXP'
    if method_name == 'DRVI-NC':
        return 'DRVI-noShare'
    # if method_name == 'TCVAE-opt':
    #     return 'B-TCVAE'
    # if method_name == 'TCVAE':
    #     return 'B-TCVAE default'
    # if method_name == 'MICHIGAN-opt':
    #     return 'MICHIGAN'
    # if method_name == 'MICHIGAN':
    #     return 'MICHIGAN default'
    if method_name == 'scvi':
        return 'scVI'
    if method_name == 'scvi-ica':
        return 'scVI-ICA'
    if method_name == 'scvi-pca':
        return 'scVI-PCA'
    if method_name == 'liger':
        return 'LIGER'
    if method_name == 'scetm':
        return 'scETM'
    if method_name == 'ica':
        return 'ICA'
    if method_name == 'mofa':
        return 'MOFA'
    if method_name == 'pca':
        return 'PCA'
    if method_name == 'poissonvi':
        return 'poissonVI'
    if method_name == 'peakvi':
        return 'peakVI'
    if method_name == 'btcvae':
        return 'B-TCVAE'
    if method_name == 'michigan':
        return 'MICHIGAN'
    return method_name


methods_general_order = [
    'DRVI', 
    'DRVI-noShare',
    'DRVI-2D',
    'DRVI-AP', 
    'DRVI-APnoEXP',
    'CVAE',
    'scETM',
    'LIGER',
    'MOFA',
    'ICA',
    'PCA',
    'scVI',
    'scVI-ICA',
    'scVI-PCA',
    'B-TCVAE',
    'MICHIGAN',
]