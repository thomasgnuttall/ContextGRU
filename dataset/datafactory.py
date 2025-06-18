from dataset.impl.bhairavi import DatasetBhairavi
from dataset.impl.saraga import DatasetSaraga


# ----------------------------------------------------------------------------------------------------------------------
class DataFactory:
    """
    A factory class for instantiating different datasets
    """
    dataset_names = [
            'sbu',
            'bhairavi',
            'saraga'
        ]

    @staticmethod
    def instantiate(dataset_name, num_synth, f, augment=False, svara_form_target=False, svara_form_group=False, root=None):
        """
        Instantiates a dataset with its name
        """

        if dataset_name not in DataFactory.dataset_names:
            raise Exception('Unknown dataset "{}"'.format(dataset_name))

        if dataset_name == "bhairavi":
            if root:
                return DatasetBhairavi(f=f, num_synth=num_synth, root=root, augment=augment, svara_form_target=svara_form_target, svara_form_group=svara_form_group)
            else:
                return DatasetBhairavi(f=f, num_synth=num_synth, augment=augment, svara_form_target=svara_form_target, svara_form_group=svara_form_group)
        
        if dataset_name == "saraga":
            return DatasetSaraga(num_synth=num_synth, augment=augment, svara_form_target=svara_form_target, svara_form_group=svara_form_group)

        raise Exception('Unknown dataset "{}"'.format(dataset_name))
