import collections
import math
import os
import struct
import statistics
import pygame
import pickle

import numpy as np
import pyaudio
from keras.engine.saving import load_model

from ml.classifier.prepare_data import preprocess_audio_chunk, HOP_LENGTH, FFT_WINDOW_SIZE
from ml.classifier.train_mobilenet import fixed_sound_length, num_mels, \
    preprocess_mobilenet_input
from ml.settings import SAMPLE_RATE, DATA_DIR

SAMPLES_PER_CHUNK = 2048
FORMAT = pyaudio.paInt16
CHANNELS = 1
RECORD_SECONDS = 5000

p = pyaudio.PyAudio()

BYTES_PER_SAMPLE = p.get_sample_size(FORMAT)
BIT_DEPTH = BYTES_PER_SAMPLE * 8

smoothing_length = 10
pygame.init()
#infoObject = pygame.display.Info()
window_width = 1450 # infoObject.current_w
window_height = 800 # infoObject.current_h
# screen = pygame.display.set_mode(FULLSCREEN)
screen = pygame.display.set_mode((window_width, window_height), pygame.FULLSCREEN)
#screen = pygame.display.set_mode((infoObject.current_w, infoObject.current_h))
running_program = True

graph_for_display = collections.deque(maxlen=int(window_width * 0.8))
for i in range(int(window_width * 0.8)):
    graph_for_display.append(0)
record_button_pos = (int(window_width * 0.1), int(window_height * 0.8))
view_button_pos = (int(window_width * 0.6), int(window_height * 0.8))

model = load_model(os.path.join(DATA_DIR / "models", "mobilenet_v2.h5"))

samples_ring_buffer = collections.deque(
    maxlen=int(math.ceil((fixed_sound_length + FFT_WINDOW_SIZE / HOP_LENGTH) * HOP_LENGTH))
    )

stream = p.open(
    format=FORMAT,
    channels=CHANNELS,
    rate=SAMPLE_RATE,
    input=True,
    frames_per_buffer=SAMPLES_PER_CHUNK,
    )

smoothing_data = collections.deque(maxlen=smoothing_length)
for i in range(smoothing_length):
    smoothing_data.append(0)

text1_pos_x = int(window_width * 0.1)
text1_pos_y = int(window_height * 0.6)
text2_pos_x = text1_pos_x
text2_pos_y = int(window_height * 0.75)
text3_pos_x = text1_pos_x
text3_pos_y = int(window_height * 0.9)
text_color = (81, 45, 45)
graph_color = (186, 24, 24)
graph_bg_color = (215, 215, 215)
screen_color = (255, 255, 255)
compare_rectangle_color = (150, 131, 48)
rectangle_color = (33, 96, 22)
highlighted_rectangle_color = (84, 165, 69)
record = []
display_cutoff = 0.4
#with open("recordings.txt", "wb") as fp:
#    pickle.dump(([[0, 0.5]], ["Rec1"]), fp)
with open("recordings.txt", "rb") as fp:
    recordings, recordings_names_text = pickle.load(fp)
compare_selection = []
is_recording = False
is_streaming = True
in_menu = False
comparing = False
writing = False
highlighted_recording = 0
headline = pygame.font.SysFont("comicsansmsXXX", int(window_height * 0.1), bold = True, italic = True)
font = pygame.font.SysFont("comicsansmsXXX", int(window_height * 0.05))
font_recordings = pygame.font.SysFont("comicsansmsXXX", int(window_height * 0.05))
recordings_names = []
for i in range(len(recordings_names_text)):
    recordings_names.append(
        font_recordings.render(recordings_names_text[i], True, text_color))
text_mirth = headline.render("Mirth-O-Meter", True, text_color)
text_record = font.render("Record (R)", True, text_color)
text_end_recording = font.render("End Recording (E)", True, text_color)
text_rec_menu = font.render("Recordings Menu (M)", True, text_color)
text_back_to_stream = font.render("Back to Stream (S)", True, text_color)
text_compare_rec = font.render("Compare selected recordings (C)", True, text_color)
text_select_recording = font.render("Select/Deselect for comparison (X)", True, text_color)
text_rename = font.render("Name/Rename (Enter)", True, text_color)
text_delete = font.render("Delete Recording (Del)", True, text_color)
new_name = ""


def integral_score(graph):
    score = 10000 * sum(graph) / len(graph)
    score = score / 10000
    return score


def threshold_score(graph, threshold):
    score = 0
    for i in range(len(graph)):
        if graph[i] > threshold:
            score += 1
    score = int(10000 * score / len(graph))
    score = score / 100
    return score

while running_program:

    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            stream.stop_stream()
            stream.close()
            p.terminate()
            running_program = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_r and is_streaming and not is_recording:
            is_recording = True
            record = []
        if event.type == pygame.KEYDOWN and event.key == pygame.K_e and is_streaming and is_recording:
            is_recording = False
            if len(recordings) < 10:
                recordings.append(record)
                recordings_names_text.append("Rec" + str(len(recordings_names)))
                recordings_names.append(
                    font_recordings.render("Rec" + str(len(recordings_names) + 1), True, text_color))
                with open("recordings.txt", "wb") as fp:
                    pickle.dump((recordings, recordings_names_text), fp)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_m and is_streaming:
            is_streaming = False
            in_menu = True
            highlighted_recording = 0
        if event.type == pygame.KEYDOWN and event.key == pygame.K_s and not writing:
            is_streaming = True
            in_menu = False
            comparing = False
        if event.type == pygame.KEYDOWN and event.key == pygame.K_c and in_menu and not writing\
                and len(compare_selection) > 0:
            in_menu = False
            comparing = True
        if event.type == pygame.KEYDOWN and event.key == pygame.K_DOWN and in_menu and not writing:
            highlighted_recording = (highlighted_recording + 1) % len(recordings)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_UP and in_menu and not writing:
            highlighted_recording = (highlighted_recording - 1) % len(recordings)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_x and in_menu and not writing:
            if highlighted_recording in compare_selection:
                compare_selection.remove(highlighted_recording)
            else:
                compare_selection.append(highlighted_recording)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_DELETE and in_menu and not writing:
            compare_selection = []
            del recordings[highlighted_recording]
            del recordings_names[highlighted_recording]
            del recordings_names_text[highlighted_recording]
            with open("recordings.txt", "wb") as fp:
                pickle.dump((recordings, recordings_names_text), fp)
        if event.type == pygame.KEYDOWN and event.key == pygame.K_RETURN and in_menu:
            if writing:
                writing = False
                recordings_names_text[highlighted_recording] = new_name
                recordings_names[highlighted_recording] = \
                    font_recordings.render(recordings_names_text[highlighted_recording], True, text_color)
                with open("recordings.txt", "wb") as fp:
                    pickle.dump((recordings, recordings_names_text), fp)
            else:
                writing = True
                new_name = ""
        if event.type == pygame.KEYDOWN and writing and not event.key == pygame.K_RETURN:
            new_name += event.unicode
        if event.type == pygame.KEYDOWN and writing and event.key == pygame.K_BACKSPACE:
            #            print(new_name)
            new_name = new_name[:-1] # not working, dunno why!
            #            print(new_name)

    data = stream.read(SAMPLES_PER_CHUNK)   # may throttle update speed. may not matter.

    for j in range(SAMPLES_PER_CHUNK):
        byte_index = j * BYTES_PER_SAMPLE
        those_bytes = data[byte_index: byte_index + BYTES_PER_SAMPLE]
        unpacked_int = struct.unpack("<h", those_bytes)[0]
        value = unpacked_int / 32767  # ends up roughly between -1 and 1
        samples_ring_buffer.append(value)

    samples = np.array(samples_ring_buffer)

    spectrogram = preprocess_audio_chunk(
        samples, fixed_sound_length=fixed_sound_length, num_mels=num_mels
        )

    x = np.array([spectrogram])
    x = preprocess_mobilenet_input(x)

    y_predicted = float(model.predict(x)[0])

    smoothing_data.append(y_predicted)
    y_smoothed = statistics.mean(smoothing_data)
    if y_smoothed < display_cutoff:
        y_smoothed = 0
    else:
        y_smoothed = (y_smoothed - display_cutoff) / (1 - display_cutoff)
    graph_for_display.append(y_smoothed ** 5)  # ** 10
    if is_recording:
        record.append(y_smoothed ** 5)   # ** 10

    screen.fill(screen_color)
    if is_streaming or in_menu:
        screen.blit(text_mirth, (int((window_width - text_mirth.get_width()) / 2), 3))
    if is_streaming:
        pygame.draw.rect(screen, graph_bg_color, pygame.Rect(int(window_width * 0.1), int(window_height * 0.1),
                                                             int(window_width * 0.8), int(window_height * 0.4)))
        for i in range(len(graph_for_display)):
            pygame.draw.rect(screen, graph_color, pygame.Rect(int(window_width * 0.1) + i,
                     (window_height * 0.5 - int(graph_for_display[i] * window_height * 0.4)), 2,
                     2 + int(graph_for_display[i] * window_height * 0.4)))
        screen.blit(text_rec_menu, (text2_pos_x, text2_pos_y))
        if is_recording:
            screen.blit(text_end_recording, (text1_pos_x, text1_pos_y))
        else:
            screen.blit(text_record, (text1_pos_x, text1_pos_y))

    elif in_menu:
        for i in range(len(recordings)):
            if i in compare_selection:
                pygame.draw.rect(screen, compare_rectangle_color,
                            pygame.Rect(int(text1_pos_x) + int(window_width / 2) * (i // 5),
                    int(window_height * 0.1) * ((i % 5) + 1),
                    int(window_width * 0.3), int(window_height * 0.09)))
            else:
                pygame.draw.rect(screen, rectangle_color,
                             pygame.Rect(int(text1_pos_x) + int(window_width / 2) * (i // 5),
                    int(window_height * 0.1) * ((i % 5) + 1),
                    int(window_width * 0.3), int(window_height * 0.09)))
            if highlighted_recording == i:
                # draws on top of blue rect
                pygame.draw.rect(screen, highlighted_rectangle_color,
                        pygame.Rect(int(text1_pos_x) + int(window_width / 2) * (i // 5),
                        int(window_height * 0.1) * ((i % 5) + 1),
                        int(window_width * 0.3), int(window_height * 0.09)))
            if highlighted_recording == i and writing:
                # draws on top of blue rect
                pygame.draw.rect(screen, screen_color,
                        pygame.Rect(int(text1_pos_x) + int(window_width / 2) * (i // 5),
                        int(window_height * 0.1) * ((i % 5) + 1),
                        int(window_width * 0.3), int(window_height * 0.09)))
            if not writing:
                screen.blit(recordings_names[i], (int(text1_pos_x) + int(window_width / 2) * (i // 5),
                    int(window_height * 0.1) * ((i % 5) + 1)))
            else:
                new_name_surface = font_recordings.render(new_name, True, text_color)
                screen.blit(new_name_surface, (int(text1_pos_x) + int(window_width / 2)
                    * (highlighted_recording // 5),
                    int(window_height * 0.1) * ((highlighted_recording % 5) + 1)))

        screen.blit(text_back_to_stream, (text1_pos_x, text1_pos_y))
        screen.blit(text_compare_rec, (text2_pos_x, text2_pos_y))
        screen.blit(text_select_recording, (text3_pos_x, text3_pos_y))
        screen.blit(text_rename, (text1_pos_x + int(window_width * 0.5), text1_pos_y))
        screen.blit(text_delete, (text2_pos_x + int(window_width * 0.5), text2_pos_y))

    elif comparing:
        longest_recording = 0
        for i in range(len(compare_selection)):
            if len(recordings[compare_selection[i]]) > longest_recording:
                longest_recording = len(recordings[compare_selection[i]])
        graph_height = int((window_height / len(compare_selection))) ####### making space for selections
        for i in range(len(compare_selection)):
            pygame.draw.rect(screen, graph_bg_color, pygame.Rect(int(0.05 * window_width),
                    int(i * graph_height),
                    int(0.9 * window_width), int(0.9 * graph_height)))
            for j in range(len(recordings[compare_selection[i]])):
                pygame.draw.rect(screen, graph_color,
                        pygame.Rect(int((0.05 + j * 0.9 / longest_recording) * window_width),
                        int((i + 0.9 - 0.9 * recordings[compare_selection[i]][j]) * graph_height), 2,
                                    int(graph_height * 0.9 * recordings[compare_selection[i]][j])))
            screen.blit(recordings_names[compare_selection[i]], (int(window_width * 0.1), int((i + 0.3) * graph_height)))
            score = threshold_score(recordings[compare_selection[i]], 0.7)
            text_score = font.render(str(score), True, text_color)
            screen.blit(text_score, (int(window_width * 0.6), int((i + 0.3) * graph_height)))

    pygame.display.flip()