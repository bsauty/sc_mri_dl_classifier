def get_modality(batch):
    labels = []
    for acq in batch['input_metadata']:
        path = acq.__getitem__('bids_metadata')['input_filename']
        name = os.path.basename(path)
        if "acq-MToff_MTS" in name :
            labels.append(0)
            continue
        if "acq-MTon_MTS" in name :
            labels.append(1)
            continue
        if "acq-T1w_MTS" in name :
            labels.append(2)
            continue
        if "T1w" in name :
            labels.append(3)
            continue
        if "T2star" in name :
            labels.append(4)
            continue
        if "T2w" in name :
            labels.append(5) 
            continue
    return labels

def OneHotEncode(labels):
    ohe_labels = []
    for label in labels :
        ohe = [0 for i in range(6)]
        ohe[label] = 1 
        ohe_labels.append(ohe)
    return torch.FloatTensor(ohe_labels)