from fedscale.dataloaders.divide_data import DataPartitioner, select_dataset
from fedscale.dataloaders.utils_data import get_data_transform
from argparse import Namespace
from torchvision import datasets, transforms
import os


class Customized_Dataloader:
    def __init__(self, dataset, args, num_class, is_test):
        self.args = args
        self.loaders = DataPartitioner(data=dataset, args=args, numOfClass=num_class, isTest=is_test)
        self.loaders.partition_data_helper(num_clients=args.num_clients,
                                           data_map_file=args.data_map_file, )
        self.internal_mapping = {}
        self._cur_index = 0
        self.__preprocess(args.batch_size)
    

    def __preprocess(self, batch_size):
        real_index = 0
        for idx, partition in enumerate(self.loaders.partitions):
            if len(partition) > batch_size or self.loaders.isTest:
                self.internal_mapping[real_index] = idx
                real_index += 1
        self._size = real_index

    def __getitem__(self, item):
        item = self.internal_mapping[item]
        return select_dataset(item+1, self.loaders, batch_size=self.args.batch_size, args=self.args, isTest=self.loaders.isTest)

    def __iter__(self):
        return self

    def __next__(self):
        if self._cur_index < self._size:
            index = self.internal_mapping[self._cur_index]
            member = select_dataset(index+1, self.loaders, batch_size=self.args.batch_size, args=self.args, isTest=self.loaders.isTest)
            self._cur_index += 1
            return member
        self._cur_index = 0
        raise StopIteration

    def __len__(self):
        return self._size


def get_loaders(data_set: str, override_args: Namespace):
    # train_dataset, test_dataset = init_dataset()
    task = "vision"
    if data_set == "speech":
        task = "speech"

    if data_set == "femnist":
        from fedscale.dataloaders.femnist import FEMNIST

        train_transform, test_transform = get_data_transform('mnist')
        train_dataset = FEMNIST(
            "data/femnist", dataset='train', transform=train_transform)
        test_dataset = FEMNIST(
            "data/femnist", dataset='test', transform=test_transform)

        args_dict = {"task": task, "batch_size": override_args.batch,
                     "num_loaders": 1, "num_clients": override_args.pd_nuser,
                     "data_map_file": "data/femnist/client_data_mapping/train.csv"}
        args = Namespace(**args_dict)
        train_loaders = Customized_Dataloader(train_dataset, args, 62, False)

        args.data_map_file = "data/femnist/client_data_mapping/test.csv"
        test_loaders = Customized_Dataloader(test_dataset, args, 62, True)

        args.data_map_file = "data/femnist/client_data_mapping/test.csv"
        val_loaders = Customized_Dataloader(test_dataset, args, 62, True)

    elif data_set == "speech":
        import numba

        from fedscale.dataloaders.speech import SPEECH, BackgroundNoiseDataset
        from fedscale.dataloaders.transforms_stft import (AddBackgroundNoiseOnSTFT,
                                                          DeleteSTFT,
                                                          FixSTFTDimension,
                                                          StretchAudioOnSTFT,
                                                          TimeshiftAudioOnSTFT,
                                                          ToMelSpectrogramFromSTFT,
                                                          ToSTFT)
        from fedscale.dataloaders.transforms_wav import (ChangeAmplitude,
                                                         ChangeSpeedAndPitchAudio,
                                                         FixAudioLength, LoadAudio,
                                                         ToMelSpectrogram,
                                                         ToTensor)
        bkg = '_background_noise_'
        data_aug_transform = transforms.Compose(
            [ChangeAmplitude(), ChangeSpeedAndPitchAudio(), FixAudioLength(), ToSTFT(), StretchAudioOnSTFT(),
             TimeshiftAudioOnSTFT(), FixSTFTDimension()])
        bg_dataset = BackgroundNoiseDataset(
            os.path.join("data/google_speech", bkg), data_aug_transform)
        add_bg_noise = AddBackgroundNoiseOnSTFT(bg_dataset)
        train_feature_transform = transforms.Compose([ToMelSpectrogramFromSTFT(
            n_mels=32), DeleteSTFT(), ToTensor('mel_spectrogram', 'input')])
        train_dataset = SPEECH("data/google_speech", dataset='train',
                               transform=transforms.Compose([LoadAudio(),
                                                             data_aug_transform,
                                                             add_bg_noise,
                                                             train_feature_transform]))
        valid_feature_transform = transforms.Compose(
            [ToMelSpectrogram(n_mels=32), ToTensor('mel_spectrogram', 'input')])
        test_dataset = SPEECH("data/google_speech", dataset='test',
                              transform=transforms.Compose([LoadAudio(),
                                                            FixAudioLength(),
                                                            valid_feature_transform]))

        args_dict = {"task": task, "batch_size": override_args.batch,
                     "num_loaders": 1, "num_clients": override_args.pd_nuser,
                     "data_map_file": "data/google_speech/client_data_mapping/train.csv"}
        args = Namespace(**args_dict)
        train_loaders = Customized_Dataloader(train_dataset, args, 12, False)

        args.data_map_file = "data/google_speech/client_data_mapping/test.csv"
        args.batch_size = 1
        test_loaders = Customized_Dataloader(test_dataset, args, 12, True)
        for test_loader in test_loaders:
            print(f"test_loader_len: {len(test_loader)}")

        args.data_map_file = "data/google_speech/client_data_mapping/test.csv"
        args.batch_size = 1
        val_loaders = Customized_Dataloader(test_dataset, args, 12, True)
    else:
        raise Exception(f"dataset {data_set} is not supported")

    return train_loaders, test_loaders, val_loaders
