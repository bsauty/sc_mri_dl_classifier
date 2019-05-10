from bids_neuropoly import bids
from medicaltorch import datasets as mt_datasets


dct_label_map = { "laacq-MToff_MTS" : 0, "acq-MTon_MTS" : 1, 
                 "acq-T1w_MTS" : 2, "T1w" : 3, "T2star" : 4, "T2w" : 5}


class BIDSSegPair2D(mt_datasets.SegmentationPair2D):
    def __init__(self, input_filename, gt_filename, metadata):
        super().__init__(input_filename, gt_filename)
        self.metadata = metadata
        self.metadata["input_filename"] = input_filename
        self.metadata["gt_filename"] = gt_filename

    def get_pair_slice(self, slice_index, slice_axis=2):
        dreturn = super().get_pair_slice(slice_index, slice_axis)
        self.metadata["slice_index"] = slice_index
        dreturn["input_metadata"]["bids_metadata"] = self.metadata
        return dreturn


class MRI2DBidsSegDataset(mt_datasets.MRI2DSegmentationDataset):
    def _load_filenames(self):
        for input_filename, gt_filename, bids_metadata in self.filename_pairs:
            segpair = BIDSSegPair2D(input_filename, gt_filename,
                                    bids_metadata)
            self.handlers.append(segpair)


class BidsDataset(MRI2DBidsSegDataset):
    def __init__(self, root_dir, slice_axis=2, cache=True,
                 transform=None, slice_filter_fn=None,
                 canonical=False, labeled=True):
        self.bids_ds = bids.BIDS(root_dir)
        self.filename_pairs = []
    
        for subject in self.bids_ds.get_subjects():

            derivatives = subject.get_derivatives("labels")
            cord_label_filename = None

            for deriv in derivatives:
                if deriv.endswith("seg-manual.nii.gz"):
                    cord_label_filename = deriv

            if cord_label_filename is None:
                continue

            metadata = subject.metadata()
            self.filename_pairs.append((subject.record.absolute_path,
                                        cord_label_filename, metadata))
            
        super().__init__(self.filename_pairs, slice_axis, cache,
                         transform, slice_filter_fn, canonical)