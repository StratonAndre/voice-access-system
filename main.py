import os
import numpy as np
import librosa
import random
import pickle
from scipy.spatial.distance import cosine
import shutil
from datetime import datetime
import time
import sounddevice as sd
import scipy.io.wavfile as wav
import speech_recognition as sr
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings("ignore", message=".*TripleDES.*")
warnings.filterwarnings("ignore", message=".*Blowfish.*")

LIVE_MODE = 1 #0 for testing with the database | 1 for live | 2 for testing with recording on the spot

DEFAULT_THRESHOLD = 0.995
RECORDING_DURATION = 120
SAMPLE_RATE = 16000
ENABLE_TRANSCRIPTION = False

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_PATH = os.path.join(SCRIPT_DIR, "storage", "audio")
LIBRISPEECH_DEV_PATH = os.path.join(BASE_PATH, "LibriSpeech", "dev-clean")
LIBRISPEECH_TEST_PATH = os.path.join(BASE_PATH, "LibriSpeech", "test-clean")
MODELS_PATH = os.path.join(BASE_PATH, "models")
FEATURES_FILE = os.path.join(MODELS_PATH, "mfcc_features.pkl")

ALLOWED_PATH = os.path.join(BASE_PATH, "allowed")
UNPROCESSED_PATH = os.path.join(BASE_PATH, "unprocessed")
ALLOWED_PROCESSED_PATH = os.path.join(BASE_PATH, "allowed_processed")
PROCESSED_PATH = os.path.join(BASE_PATH, "processed")
TEMP_RECORDING_PATH = os.path.join(BASE_PATH, "temp_recording.wav")

for path in [MODELS_PATH, ALLOWED_PATH, UNPROCESSED_PATH, ALLOWED_PROCESSED_PATH, PROCESSED_PATH]:
    os.makedirs(path, exist_ok=True)


def extract_mfcc_features(audio_path, n_mfcc=13, n_fft=2048, hop_length=512):
    try:
        y, sr = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=n_fft, hop_length=hop_length)

        features = []
        features.extend(np.mean(mfccs, axis=1))
        features.extend(np.std(mfccs, axis=1))
        features.extend(np.max(mfccs, axis=1))
        features.extend(np.min(mfccs, axis=1))

        delta_mfccs = librosa.feature.delta(mfccs)
        delta2_mfccs = librosa.feature.delta(mfccs, order=2)

        features.extend(np.mean(delta_mfccs, axis=1))
        features.extend(np.mean(delta2_mfccs, axis=1))

        return np.array(features)

    except:
        return None


def create_speaker_model(audio_files):
    features = []

    for audio_file in audio_files:
        feature = extract_mfcc_features(audio_file)
        if feature is not None:
            features.append(feature)

    if len(features) == 0:
        raise ValueError("No features could be extracted")

    return np.mean(features, axis=0)


def compare_speakers(model1, model2):
    return 1 - cosine(model1, model2)


def verify_speaker(speaker_models, claimed_id, test_recording):
    if claimed_id not in speaker_models:
        return False, 0.0

    test_features = extract_mfcc_features(test_recording)
    if test_features is None:
        return False, 0.0

    similarity = compare_speakers(speaker_models[claimed_id], test_features)

    return similarity > DEFAULT_THRESHOLD, similarity


def is_valid_audio_file(filename):
    return filename.endswith(('.flac', '.wav', '.mp3'))


def verify_audio_file(audio_path):
    try:
        with sr.AudioFile(audio_path) as source:

            recognizer = sr.Recognizer()
            recognizer.record(source, duration=0.1)
        return True
    except Exception as e:
        print(f"Audio verification failed: {e}")
        return False


def parse_speaker_id(filename):
    try:
        parts = filename.split('-')
        if len(parts) >= 3:
            return int(parts[0])
    except:
        pass
    return None


def collect_all_speaker_recordings():
    all_recordings = {}

    for dataset_path in [LIBRISPEECH_DEV_PATH, LIBRISPEECH_TEST_PATH]:
        for root, _, files in os.walk(dataset_path):
            for file in files:
                if is_valid_audio_file(file):
                    speaker_id = parse_speaker_id(file)
                    if speaker_id is not None:
                        if speaker_id not in all_recordings:
                            all_recordings[speaker_id] = []
                        all_recordings[speaker_id].append(os.path.join(root, file))

    return all_recordings


def build_authorized_speaker_models():
    speaker_models = {}
    speaker_files = {}

    for filename in os.listdir(ALLOWED_PATH):
        if is_valid_audio_file(filename):
            filepath = os.path.join(ALLOWED_PATH, filename)
            base_name = os.path.splitext(filename)[0]

            if '-' in base_name:
                try:
                    speaker_id = base_name.split('-')[0]
                except:
                    speaker_id = base_name
            else:
                speaker_id = base_name

            if speaker_id not in speaker_files:
                speaker_files[speaker_id] = []
            speaker_files[speaker_id].append(filepath)

    for speaker_id, files in speaker_files.items():
        try:
            speaker_models[speaker_id] = create_speaker_model(files)
            print(f"Created model for speaker: {speaker_id} (using {len(files)} file(s))")
        except Exception as e:
            print(f"Failed to create model for {speaker_id}: {e}")

    return speaker_models


def manage_processed_files(directory, max_files=20):
    files = []
    for filename in os.listdir(directory):
        filepath = os.path.join(directory, filename)
        if os.path.isfile(filepath):
            files.append((filepath, os.path.getmtime(filepath)))

    if len(files) > max_files:
        files.sort(key=lambda x: x[1])

        files_to_remove = len(files) - max_files
        for i in range(files_to_remove):
            os.remove(files[i][0])
            print(f"Removed old file: {os.path.basename(files[i][0])}")


def record_audio(duration=RECORDING_DURATION):
    print(f"Recording for {duration} seconds...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='int16')
    sd.wait()

    if recording.dtype != np.int16:
        recording = (recording * 32767).astype(np.int16)

    wav.write(TEMP_RECORDING_PATH, SAMPLE_RATE, recording)
    print("Recording completed")
    return TEMP_RECORDING_PATH


def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    try:
        print(f"Attempting to transcribe: {audio_path}")

        try:
            with sr.AudioFile(audio_path) as source:
                audio = recognizer.record(source)
                text = recognizer.recognize_google(audio)
                return text
        except Exception as e:
            print(f"Primary transcription failed: {e}")

            try:
                print("Attempting fallback transcription method...")
                y, sr_rate = librosa.load(audio_path, sr=16000)

                temp_wav = audio_path + ".temp.wav"
                audio_int16 = (y * 32767).astype(np.int16)
                wav.write(temp_wav, 16000, audio_int16)

                with sr.AudioFile(temp_wav) as source:
                    audio = recognizer.record(source)
                    text = recognizer.recognize_google(audio)

                os.remove(temp_wav)
                return text

            except Exception as e2:
                print(f"Fallback transcription also failed: {e2}")
                return "[Failed to transcribe]"

    except sr.UnknownValueError:
        print("Google Speech Recognition could not understand audio")
        return "[Unintelligible]"
    except sr.RequestError as e:
        print(f"Could not request results from Google Speech Recognition service; {e}")
        return "[Error]"
    except Exception as e:
        print(f"Speech recognition error: {e}")
        return "[Failed to transcribe]"


def get_next_speaker_id():
    existing_ids = []

    for filename in os.listdir(ALLOWED_PATH):
        if is_valid_audio_file(filename):
            base_name = os.path.splitext(filename)[0]
            if '-' in base_name:
                try:
                    speaker_id = int(base_name.split('-')[0])
                    existing_ids.append(speaker_id)
                except ValueError:
                    continue

    if existing_ids:
        next_id = max(existing_ids) + 1
    else:
        next_id = 1001

    return next_id


def add_new_authorized_speaker():

    print("\n--- Adding New Authorized Speaker ---")

    speaker_id = get_next_speaker_id()

    print(f"Generated Speaker ID: {speaker_id}")
    print("Press Enter when ready to record (10 seconds)")
    print("Say something to identify yourself (e.g., 'This is my voice sample')")
    input()

    audio_path = record_audio(duration=10)

    filename = f"{speaker_id}-0-0001.wav"
    dest_path = os.path.join(ALLOWED_PATH, filename)

    try:
        y, sr_rate = librosa.load(audio_path, sr=16000)
        audio_int16 = (y * 32767).astype(np.int16)
        wav.write(dest_path, 16000, audio_int16)
        os.remove(audio_path)
        print("Audio file saved successfully")
    except Exception as e:
        print(f"Warning: Could not optimize audio format: {e}")
        shutil.move(audio_path, dest_path)

    print(f"Added speaker with ID: {speaker_id}")

    if ENABLE_TRANSCRIPTION:
        text = transcribe_audio(dest_path)
        print(f"Transcribed enrollment text: {text}")

    trans_filename = f"{speaker_id}-0.trans.txt"
    trans_path = os.path.join(ALLOWED_PATH, trans_filename)
    with open(trans_path, 'w') as f:
        f.write(f"{speaker_id}-0-0001 {text if ENABLE_TRANSCRIPTION else 'Voice enrollment sample'}\n")

    return str(speaker_id)


def add_authorized_speaker_multiple_samples():

    print("\n--- Adding New Authorized Speaker (Multiple Samples) ---")

    speaker_id = get_next_speaker_id()

    print(f"Generated Speaker ID: {speaker_id}")

    num_samples = 3
    print(f"We'll record {num_samples} samples for better accuracy")

    for i in range(num_samples):
        print(f"\nSample {i + 1}/{num_samples}")
        print("Press Enter when ready to record (10 seconds)")
        print("Say something different each time (e.g., count numbers, say alphabet, or speak naturally)")
        input()

        audio_path = record_audio(duration=10)

        utterance_id = f"{i + 1:04d}"
        filename = f"{speaker_id}-0-{utterance_id}.wav"
        dest_path = os.path.join(ALLOWED_PATH, filename)
        shutil.move(audio_path, dest_path)

        print(f"Saved sample {i + 1}")

        if ENABLE_TRANSCRIPTION:
            text = transcribe_audio(dest_path)
            print(f"Transcribed text: {text}")

            trans_filename = f"{speaker_id}-0.trans.txt"
            trans_path = os.path.join(ALLOWED_PATH, trans_filename)
            with open(trans_path, 'a') as f:
                f.write(f"{speaker_id}-0-{utterance_id} {text}\n")

    print(f"\nAdded speaker with ID: {speaker_id} using {num_samples} samples")
    return str(speaker_id)


def list_authorized_speakers():
    """List all authorized speakers"""
    speaker_info = {}

    for filename in os.listdir(ALLOWED_PATH):
        if is_valid_audio_file(filename):
            base_name = os.path.splitext(filename)[0]

            if '-' in base_name:
                try:
                    speaker_id = base_name.split('-')[0]
                except:
                    speaker_id = base_name
            else:
                speaker_id = base_name

            if speaker_id not in speaker_info:
                speaker_info[speaker_id] = 0
            speaker_info[speaker_id] += 1

    trans_files = {}
    for filename in os.listdir(ALLOWED_PATH):
        if filename.endswith('.trans.txt'):
            speaker_id = filename.split('-')[0]
            trans_path = os.path.join(ALLOWED_PATH, filename)
            try:
                with open(trans_path, 'r') as f:
                    trans_files[speaker_id] = len(f.readlines())
            except:
                pass

    print("\n--- Authorized Speakers ---")
    if speaker_info:
        for speaker_id, count in sorted(speaker_info.items()):
            trans_count = trans_files.get(speaker_id, 0)
            print(f"Speaker ID: {speaker_id} - {count} audio file(s), {trans_count} transcript(s)")
    else:
        print("No authorized speakers found")
    print()


def process_microphone_recording(speaker_models):
    audio_path = record_audio()

    authorized = False
    matched_speaker = None
    best_similarity = 0.0

    for speaker_id in speaker_models:
        is_match, similarity = verify_speaker(speaker_models, speaker_id, audio_path)
        if is_match and similarity > best_similarity:
            authorized = True
            matched_speaker = speaker_id
            best_similarity = similarity

    if authorized:
        print(f"Access granted for {matched_speaker} (similarity: {best_similarity:.3f})")

        text = transcribe_audio(audio_path)
        print(f"Transcribed text: {text}")

        if "open door" in text.lower() or "door open" in text.lower():
            print("THE DOOR IS OPEN")

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        new_filename = f"{matched_speaker}_{timestamp}.wav"
        dest_path = os.path.join(PROCESSED_PATH, new_filename)
        shutil.move(audio_path, dest_path)
        manage_processed_files(PROCESSED_PATH, max_files=20)

        return True, matched_speaker, text
    else:
        print("Access denied. Recording deleted.")
        os.remove(audio_path)
        return False, None, None


def microphone_test_mode():
    print("Microphone Test Mode (LIVE_MODE = 2)")
    print("Press 1 to record for 2 minutes")
    print("Press 2 for quick test (10 seconds)")
    print("Press 3 to add new authorized speaker (single sample)")
    print("Press 4 to add new authorized speaker (multiple samples)")
    print("Press 5 to list authorized speakers")
    print("Press q to quit")

    speaker_models = build_authorized_speaker_models()

    if not speaker_models:
        print("\nNo authorized speaker models found!")
        print("You need to add authorized speakers first.")
        print("Use option 3 to add new authorized speakers.\n")
    else:
        print(f"\nLoaded {len(speaker_models)} authorized speaker models")
        print(f"Authorized speakers: {list(speaker_models.keys())}\n")

    while True:
        choice = input("\nYour choice: ")

        if choice == '1':
            if not speaker_models:
                print("No authorized speakers available. Please add speakers first (option 3).")
                continue
            process_microphone_recording(speaker_models)

        elif choice == '2':
            if not speaker_models:
                print("No authorized speakers available. Please add speakers first (option 3).")
                continue

            audio_path = record_audio(duration=10)

            authorized = False
            matched_speaker = None
            best_similarity = 0.0

            for speaker_id in speaker_models:
                is_match, similarity = verify_speaker(speaker_models, speaker_id, audio_path)
                if is_match and similarity > best_similarity:
                    authorized = True
                    matched_speaker = speaker_id
                    best_similarity = similarity

            if authorized:
                print(f"Quick test: Access granted for {matched_speaker} (similarity: {best_similarity:.3f})")
                text = transcribe_audio(audio_path)
                print(f"Text: {text}")
                if "open door" in text.lower() or "door open" in text.lower():
                    print("THE DOOR IS OPEN")
            else:
                print("Quick test: Access denied")

            os.remove(audio_path)

        elif choice == '3':
            new_speaker = add_new_authorized_speaker()
            if new_speaker:
                speaker_models = build_authorized_speaker_models()
                print(f"\nUpdated speaker models. Total authorized speakers: {len(speaker_models)}")

        elif choice == '4':
            new_speaker = add_authorized_speaker_multiple_samples()
            if new_speaker:
                speaker_models = build_authorized_speaker_models()
                print(f"\nUpdated speaker models. Total authorized speakers: {len(speaker_models)}")

        elif choice == '5':
            list_authorized_speakers()

        elif choice == 'q':
            break
        else:
            print("Invalid choice")


def live_mode_processing():
    print("Starting Live Mode Processing (LIVE_MODE = 1)...")

    speaker_models = build_authorized_speaker_models()

    if not speaker_models:
        print("No authorized speaker models found! Add audio files to the 'allowed' directory.")
        return

    print(f"Loaded {len(speaker_models)} authorized speaker models")

    while True:
        unprocessed_files = [f for f in os.listdir(UNPROCESSED_PATH) if is_valid_audio_file(f)]

        for filename in unprocessed_files:
            filepath = os.path.join(UNPROCESSED_PATH, filename)
            print(f"\nProcessing: {filename}")

            authorized = False
            matched_speaker = None
            best_similarity = 0.0

            for speaker_id in speaker_models:
                is_match, similarity = verify_speaker(speaker_models, speaker_id, filepath)
                if is_match and similarity > best_similarity:
                    authorized = True
                    matched_speaker = speaker_id
                    best_similarity = similarity

            if authorized:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                new_filename = f"{matched_speaker}_{timestamp}_{filename}"
                dest_path = os.path.join(PROCESSED_PATH, new_filename)
                shutil.move(filepath, dest_path)
                print(f"Access granted for {matched_speaker} (similarity: {best_similarity:.3f})")
                print(f"Moved to: {dest_path}")

                manage_processed_files(PROCESSED_PATH, max_files=20)
            else:
                os.remove(filepath)
                print(f"Access denied. File deleted: {filename}")

        time.sleep(1)


def process_allowed_files():
    print("Processing allowed files...")

    for filename in os.listdir(ALLOWED_PATH):
        if is_valid_audio_file(filename):
            src_path = os.path.join(ALLOWED_PATH, filename)

            features = extract_mfcc_features(src_path)
            if features is not None:
                dest_path = os.path.join(ALLOWED_PROCESSED_PATH, filename)
                shutil.copy2(src_path, dest_path)
                print(f"Processed allowed file: {filename}")
            else:
                print(f"Failed to process allowed file: {filename}")


def main():
    if LIVE_MODE == 2:
        microphone_test_mode()
    elif LIVE_MODE == 1:
        process_allowed_files()
        live_mode_processing()
    else:
        all_speakers = collect_all_speaker_recordings()

        speaker_list = list(all_speakers.keys())
        random.shuffle(speaker_list)

        authorized_ids = speaker_list[:5]

        speaker_models = {}
        for speaker_id in authorized_ids:
            recordings = all_speakers[speaker_id]
            train_size = max(1, int(len(recordings) * 0.8))
            train_recordings = recordings[:train_size]

            try:
                speaker_models[speaker_id] = create_speaker_model(train_recordings)
            except:
                continue

        with open(FEATURES_FILE, 'wb') as f:
            pickle.dump(speaker_models, f)

        auth_granted = 0
        auth_denied = 0

        for speaker_id in authorized_ids:
            recordings = all_speakers[speaker_id]
            train_size = max(1, int(len(recordings) * 0.8))
            test_recordings = recordings[train_size:]

            if len(test_recordings) == 0:
                test_recordings = [recordings[-1]]

            for test_recording in test_recordings:
                try:
                    is_match, similarity = verify_speaker(speaker_models, speaker_id, test_recording)

                    if is_match:
                        auth_granted += 1
                        if ENABLE_TRANSCRIPTION:
                            print(f"\n[GRANTED] Speaker {speaker_id} (similarity: {similarity:.3f})")
                            print(f"Audio file: {test_recording}")
                            text = transcribe_audio(test_recording)
                            print(f"Transcribed text: {text}")
                            if text and ("open door" in text.lower() or "door open" in text.lower()):
                                print(">>> THE DOOR IS OPEN <<<")
                    else:
                        auth_denied += 1
                except Exception as e:
                    if ENABLE_TRANSCRIPTION:
                        print(f"Error processing {test_recording}: {e}")
                    continue

        unauth_granted = 0
        unauth_denied = 0

        unauthorized_ids = [id for id in all_speakers.keys() if id not in authorized_ids]

        for speaker_id in unauthorized_ids:
            recordings = all_speakers[speaker_id]

            for test_recording in recordings:
                claimed_id = random.choice(authorized_ids)

                try:
                    is_match, _ = verify_speaker(speaker_models, claimed_id, test_recording)

                    if is_match:
                        unauth_granted += 1
                    else:
                        unauth_denied += 1
                except:
                    continue

        total_auth_tests = auth_granted + auth_denied
        total_unauth_tests = unauth_granted + unauth_denied

        print(f"Authorized users: {authorized_ids}")
        print(f"Total speakers: {len(all_speakers)}")
        print(f"Threshold: {DEFAULT_THRESHOLD}")
        print()
        print("Authorized users:")
        print(f"  Tests: {total_auth_tests}")
        print(f"  Granted: {auth_granted}")
        print(f"  Denied: {auth_denied}")
        if total_auth_tests > 0:
            print(f"  True Accept Rate: {auth_granted / total_auth_tests * 100:.1f}%")

        print()
        print("Unauthorized users:")
        print(f"  Tests: {total_unauth_tests}")
        print(f"  Granted: {unauth_granted}")
        print(f"  Denied: {unauth_denied}")
        if total_unauth_tests > 0:
            print(f"  False Accept Rate: {unauth_granted / total_unauth_tests * 100:.1f}%")

        print()
        print(f"Total tests: {total_auth_tests + total_unauth_tests}")


if __name__ == "__main__":
    main()