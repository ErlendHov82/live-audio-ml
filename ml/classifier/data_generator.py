import os
import random
import uuid
from pathlib import Path

import math
import numpy as np
from PIL import Image
from audiomentations import Compose, AddGaussianNoise, TimeStretch, PitchShift, Shift
from keras.utils import Sequence
from scipy.io import wavfile

from ml.classifier.categories import (
    CATEGORIES,
    is_laughter_category,
)
from ml.classifier.prepare_data import (
    load_wav_file,
    preprocess_audio_chunk,
    FIXED_SOUND_LENGTH,
    NUM_MELS,
    HOP_LENGTH)
from ml.settings import (
    AUDIO_EVENT_DATASET_PATH,
    SAMPLE_RATE,
    CUSTOM_AUDIO_SET_DATA_PATH_LAUGHTER,
    CUSTOM_AUDIO_SET_DATA_PATH_NOT_LAUGHTER,
)
from ml.utils.filename import get_file_paths

LAUGHTER_CLASS_RATIO = 0.3
CUT_NON_LAUGHTER_RATIO = 0.5
LOOP_RATIO = 1
SILENCE_RATIO = 1


def write_wav_file(sound_np, output_file_path, sample_rate):
    wavfile.write(output_file_path, sample_rate, sound_np)

def get_train_paths():
    sound_file_paths = {}
    for category in CATEGORIES:
        sound_file_paths[category] = get_file_paths(
            AUDIO_EVENT_DATASET_PATH / "train" / category
        )

    # We disable the laughter category of the original dataset and instead use just the
    # custom dataset, which includes some hand-curated examples from the original dataset.
    sound_file_paths["laughter"] = []

    # get paths from custom dataset in addition to the freesound audio event dataset
    custom_laughter_paths = get_file_paths(CUSTOM_AUDIO_SET_DATA_PATH_LAUGHTER)
    sound_file_paths["laughter"] += custom_laughter_paths

    custom_non_laughter_paths = get_file_paths(CUSTOM_AUDIO_SET_DATA_PATH_NOT_LAUGHTER)
    sound_file_paths["custom_non_laughter"] = custom_non_laughter_paths

    return sound_file_paths


def get_validation_paths():
    sound_file_paths = {}
    for category in CATEGORIES:
        sound_file_paths[category] = get_file_paths(
            AUDIO_EVENT_DATASET_PATH / "test", filename_prefix=category
        )

    return sound_file_paths


class SoundExampleGenerator(Sequence):
    def __init__(
        self,
        sound_file_paths,
        batch_size=8,
        augment=True,
        save_augmented_images_to_path=None,
        save_augmented_sounds_to_path=None,
        fixed_sound_length=FIXED_SOUND_LENGTH,
        num_mels=NUM_MELS,
        preprocessing_fn=None,
    ):
        self.sound_file_paths = sound_file_paths
        self.batch_size = batch_size
        self.augment = augment
        self.save_augmented_images_to_path = save_augmented_images_to_path
        self.save_augmented_sounds_to_path = save_augmented_sounds_to_path
        self.fixed_sound_length = fixed_sound_length
        self.min_num_samples = (fixed_sound_length + 3) * HOP_LENGTH
        self.num_mels = num_mels
        self.preprocessing_fn = preprocessing_fn

        self.laughter_paths = self.sound_file_paths["laughter"]
        self.non_laughter_paths = []
        for category in self.sound_file_paths:
            if not is_laughter_category(category):
                self.non_laughter_paths += self.sound_file_paths[category]

        if save_augmented_images_to_path:
            os.makedirs(save_augmented_images_to_path, exist_ok=True)

        self.augmenter = Compose(
            [
                AddGaussianNoise(min_amplitude=0.001, max_amplitude=0.002, p=0.1),
                TimeStretch(min_rate=0.8, max_rate=1.25, p=0.02),
                PitchShift(min_semitones=-3, max_semitones=3, p=0.02),
                Shift(min_fraction=-0.5, max_fraction=0.5, p=0.5),
            ]
        )

    def __len__(self):
        return math.ceil(len(self.sound_file_paths) / self.batch_size)

    def __getitem__(self, idx):
        # Note: The returned batch does not depend on idx at the moment
        x = []
        y = []
        for _ in range(self.batch_size):
            cut_non_laughter = False

            if random.random() < LAUGHTER_CLASS_RATIO:
                sound_file_path = random.choice(self.laughter_paths)
                target = 1  # laughter

            else:
                sound_file_path = random.choice(self.non_laughter_paths)
                target = 0  # not laughter
                if random.random() < CUT_NON_LAUGHTER_RATIO:
                    cut_non_laughter = True

            sound_np = load_wav_file(sound_file_path)

            if self.augment:
                sound_np = self.augmenter(samples=sound_np, sample_rate=SAMPLE_RATE)

            if cut_non_laughter:
                seconds_remaining = 1 + random.random() * 3
                num_samples_to_keep = int(seconds_remaining * SAMPLE_RATE) # rounds off?
                start_cut = int(random.random() * (len(sound_np) - num_samples_to_keep))
                sound_np = sound_np[start_cut:start_cut+num_samples_to_keep]

            if random.random() < LOOP_RATIO:
                # Repeat the sound if it is too small to fill the expected spectrogram window
                while len(sound_np) < self.min_num_samples:
                    sound_np = np.concatenate((sound_np, sound_np))
            elif random.random() > SILENCE_RATIO:
                if len(sound_np) < self.min_num_samples:
                    background_sound_file_path = random.choice(self.non_laughter_paths)
                    background_sound_np = load_wav_file(background_sound_file_path)
                    background_sound_np[-len(sound_np):] += sound_np
                    sound_np = background_sound_np

            spectrogram = preprocess_audio_chunk(
                sound_np,
                fixed_sound_length=self.fixed_sound_length,
                num_mels=self.num_mels,
            )
            if self.save_augmented_images_to_path:
                # Save the augmented image(vectors) to path
                generated_uuid = uuid.uuid4()
                input_image_pil = Image.fromarray((spectrogram * 255).astype(np.uint8))
                input_image_pil.save(
                    os.path.join(
                        self.save_augmented_images_to_path,
                        "{}__{}_input.png".format(
                            Path(sound_file_path).stem, generated_uuid
                        ),
                    )
                )

            if self.save_augmented_sounds_to_path:
                # Save the augmented sound to path
                generated_uuid = uuid.uuid4()
                output_file_path = os.path.join(
                        self.save_augmented_sounds_to_path,
                        "{}__{}_input.wav".format(
                            Path(sound_file_path).stem, generated_uuid
                        ),
                    )
                write_wav_file(sound_np, output_file_path, SAMPLE_RATE)

            x.append(spectrogram)
            y.append(target)

        x = np.array(x)
        y = np.array(y)

        if self.preprocessing_fn:
            x = self.preprocessing_fn(x)

        return x, y
