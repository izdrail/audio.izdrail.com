#!/usr/bin/env python3
"""
Enhanced Zodiac Signs Horoscope Video Generator
Creates videos with background images, better typography, and synchronized audio
Fixed version with proper folder organization and error handling
"""

import requests
import json
import os
import re
import glob
import random
import shutil
from datetime import datetime
from typing import List, Dict, Optional, Tuple
from gradio_client import Client
from moviepy.editor import AudioFileClip, ImageClip, CompositeVideoClip, concatenate_videoclips
from PIL import Image, ImageDraw, ImageFont, ImageEnhance, ImageFilter
import librosa
import numpy as np
from textwrap import wrap
import platform


class EnhancedHoroscopeGenerator:
    """Enhanced class for horoscope video generation with background images and better typography"""

    def __init__(self, tts_url: str = "http://localhost:1602/"):
        self.base_url = "https://horoscope-app-api.vercel.app/api/v1"
        self.tts_url = tts_url
        self.zodiac_signs = [
            "Aries", "Taurus", "Gemini", "Cancer", "Leo", "Virgo",
            "Libra", "Scorpio", "Sagittarius", "Capricorn", "Aquarius", "Pisces"
        ]
        self.available_voices = [
            "Standard Voice (Non-Cloned)",
            "stefan-3d785da5",
            "stefan-ebbf6485",
            "stefan-a3164eb6",
            "iliescu-ba91453c",
            "churchil-3b156ef9",
            "cris-af070b0a",
            "dsadsadsa-1d928aff",
            "stefan2222-9eca2e",
            "combined-2-voices-8cfc95",
            "gabriel-b01c30",
            "combinedsexyvoices-d97109",
            "asmrvoice-42f215",
            "greatest_british_accent-baedc3"
        ]

        # Zodiac sign colors (fallback if no background image)
        self.zodiac_colors = {
            "Aries": (255, 107, 107),
            "Taurus": (78, 205, 196),
            "Gemini": (69, 183, 209),
            "Cancer": (150, 206, 180),
            "Leo": (255, 234, 167),
            "Virgo": (221, 160, 221),
            "Libra": (152, 216, 200),
            "Scorpio": (247, 220, 111),
            "Sagittarius": (187, 143, 206),
            "Capricorn": (133, 193, 233),
            "Aquarius": (248, 196, 113),
            "Pisces": (130, 224, 170)
        }

        # Initialize directories
        self.output_dir = "output"
        self.temp_dir = "temp"
        self.images_dir = "images"
        self._create_base_directories()

        # Font paths for different operating systems
        self.font_paths = self._get_font_paths()
        self.available_fonts = self._discover_fonts()

        self.tts_client = None
        self._init_tts_client()

    def _create_base_directories(self):
        """Create base directories for organized file storage"""
        directories = [
            self.output_dir,
            self.temp_dir,
            self.images_dir,
            os.path.join(self.images_dir, "zodiac_chosen")
        ]

        for directory in directories:
            os.makedirs(directory, exist_ok=True)

    def _get_font_paths(self) -> List[str]:
        """Get font paths for different operating systems"""
        system = platform.system()
        font_paths = []

        if system == "Windows":
            font_paths = [
                "C:/Windows/Fonts/",
                "C:/Windows/System32/fonts/"
            ]
        elif system == "Darwin":  # macOS
            font_paths = [
                "/System/Library/Fonts/",
                "/Library/Fonts/",
                os.path.expanduser("~/Library/Fonts/")
            ]
        elif system == "Linux":
            font_paths = [
                "/usr/share/fonts/",
                "/usr/local/share/fonts/",
                os.path.expanduser("~/.fonts/"),
                os.path.expanduser("~/.local/share/fonts/")
            ]

        return font_paths

    def _discover_fonts(self) -> List[str]:
        """Discover available fonts on the system"""
        common_fonts = [
            # Windows fonts
            "arial.ttf", "arialbd.ttf", "calibri.ttf", "calibrib.ttf",
            "times.ttf", "timesbd.ttf", "verdana.ttf", "verdanab.ttf",
            "comic.ttf", "comicbd.ttf", "impact.ttf", "tahoma.ttf",
            "trebuc.ttf", "trebucbd.ttf", "georgia.ttf", "georgiab.ttf",

            # macOS fonts
            "Arial.ttf", "Arial Bold.ttf", "Helvetica.ttc", "Times New Roman.ttf",
            "Verdana.ttf", "Georgia.ttf", "Trebuchet MS.ttf", "Impact.ttf",
            "Comic Sans MS.ttf", "Tahoma.ttf", "Palatino.ttc", "Futura.ttc",

            # Linux fonts
            "DejaVuSans.ttf", "DejaVuSans-Bold.ttf", "DejaVuSans-Oblique.ttf",
            "Liberation Sans.ttf", "Liberation Serif.ttf", "Ubuntu-R.ttf", "Ubuntu-B.ttf"
        ]

        available_fonts = []

        for font_path in self.font_paths:
            if os.path.exists(font_path):
                for font_name in common_fonts:
                    full_path = os.path.join(font_path, font_name)
                    if os.path.exists(full_path):
                        available_fonts.append(full_path)

                # Also search for any .ttf files in the directory
                try:
                    for font_file in glob.glob(os.path.join(font_path, "*.ttf")):
                        if font_file not in available_fonts:
                            available_fonts.append(font_file)
                    for font_file in glob.glob(os.path.join(font_path, "*.ttc")):
                        if font_file not in available_fonts:
                            available_fonts.append(font_file)
                except:
                    pass

        return available_fonts

    def _init_tts_client(self):
        """Initialize the TTS client"""
        try:
            self.tts_client = Client(self.tts_url)
            print("TTS client initialized successfully")
        except Exception as e:
            print(f"Warning: Could not initialize TTS client: {e}")
            print("Audio generation will not be available")

    def generate_audio(self, text: str, speaker_id: str = "Standard Voice (Non-Cloned)",
                       save_audio: bool = True) -> Optional[tuple]:
        """Generate audio from text using TTS"""
        if not self.tts_client:
            print("TTS client not initialized. Cannot generate audio.")
            return None

        try:
            result = self.tts_client.predict(
                text=text,
                speaker_id=speaker_id,
                save_audio=save_audio,
                api_name="/tts"
            )
            return result
        except Exception as e:
            print(f"Error generating audio: {e}")
            return None

    def split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for better synchronization"""
        sentences = re.split(r'(?<=[.!?])\s+', text)
        sentences = [s.strip() for s in sentences if s.strip()]
        return sentences

    def estimate_sentence_durations(self, audio_path: str, sentences: List[str]) -> List[Tuple[float, float]]:
        """Estimate duration for each sentence based on audio length and text length"""
        try:
            audio_clip = AudioFileClip(audio_path)
            total_duration = audio_clip.duration
            audio_clip.close()

            total_chars = sum(len(s) for s in sentences)
            if total_chars == 0:
                return [(0, total_duration)]

            durations = []
            current_time = 0

            for i, sentence in enumerate(sentences):
                sentence_duration = (len(sentence) / total_chars) * total_duration
                sentence_duration = max(1.5, sentence_duration)

                end_time = min(current_time + sentence_duration, total_duration)
                durations.append((current_time, end_time))
                current_time = end_time

            if durations:
                durations[-1] = (durations[-1][0], total_duration)

            return durations

        except Exception as e:
            print(f"Error estimating sentence durations: {e}")
            return [(0, 10)]  # Fallback

    def get_background_image(self, sign: str, size: Tuple[int, int] = (1280, 720)) -> Optional[Image.Image]:
        """Get background image for the zodiac sign"""
        image_folder = os.path.join(self.images_dir, "zodiac_chosen", sign.lower())

        if not os.path.exists(image_folder):
            image_folder = os.path.join(self.images_dir, "zodiac_chosen", sign.title())

        if os.path.exists(image_folder):
            image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.gif']
            image_files = []

            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(image_folder, ext)))
                image_files.extend(glob.glob(os.path.join(image_folder, ext.upper())))

            if image_files:
                selected_image = random.choice(image_files)
                try:
                    img = Image.open(selected_image)
                    if img.mode != 'RGB':
                        img = img.convert('RGB')

                    img = img.resize(size, Image.Resampling.LANCZOS)
                    enhancer = ImageEnhance.Brightness(img)
                    img = enhancer.enhance(0.6)
                    img = img.filter(ImageFilter.GaussianBlur(radius=1))
                    return img
                except Exception as e:
                    print(f"Error loading background image {selected_image}: {e}")

        return None

    def get_best_font(self, text: str, max_width: int, max_height: int, min_size: int = 24) -> Tuple[
        Optional[ImageFont.ImageFont], int]:
        """Get the best font and size for the given text and constraints"""
        if not self.available_fonts:
            return None, min_size

        best_font = None
        best_size = min_size

        for font_path in self.available_fonts[:5]:
            try:
                for size in range(72, min_size - 1, -4):
                    font = ImageFont.truetype(font_path, size)
                    dummy_img = Image.new('RGB', (max_width, max_height))
                    dummy_draw = ImageDraw.Draw(dummy_img)
                    chars_per_line = max_width // (size // 2)
                    wrapped_text = '\n'.join(wrap(text, chars_per_line))

                    bbox = dummy_draw.textbbox((0, 0), wrapped_text, font=font)
                    text_width = bbox[2] - bbox[0]
                    text_height = bbox[3] - bbox[1]

                    if text_width <= max_width * 0.9 and text_height <= max_height * 0.6:
                        if size > best_size:
                            best_font = font
                            best_size = size
                        break

            except Exception as e:
                continue

        return best_font, best_size

    def create_text_image(self, text: str, sign: str, size: Tuple[int, int] = (1280, 720)) -> str:
        """Create an image with text overlay and background image"""
        background = self.get_background_image(sign, size)

        if background:
            img = background.copy()
        else:
            bg_color = self.zodiac_colors.get(sign, (74, 144, 226))
            img = Image.new('RGB', size, bg_color)

        draw = ImageDraw.Draw(img)
        margin = 60
        text_width = size[0] - (margin * 2)
        text_height = size[1] - (margin * 3)

        main_font, main_size = self.get_best_font(text, text_width, text_height, min_size=32)

        if main_font:
            chars_per_line = text_width // (main_size // 2)
        else:
            chars_per_line = 50
            main_size = 32

        wrapped_text = '\n'.join(wrap(text, chars_per_line))

        if main_font:
            bbox = draw.textbbox((0, 0), wrapped_text, font=main_font)
            text_width_actual = bbox[2] - bbox[0]
            text_height_actual = bbox[3] - bbox[1]
        else:
            lines = wrapped_text.split('\n')
            text_width_actual = max(len(line) for line in lines) * (main_size // 2)
            text_height_actual = len(lines) * (main_size + 4)

        text_x = (size[0] - text_width_actual) // 2
        text_y = (size[1] - text_height_actual) // 2 + 30

        overlay = Image.new('RGBA', size, (0, 0, 0, 0))
        overlay_draw = ImageDraw.Draw(overlay)

        padding = 20
        rect_coords = [
            text_x - padding,
            text_y - padding,
            text_x + text_width_actual + padding,
            text_y + text_height_actual + padding
        ]
        overlay_draw.rounded_rectangle(rect_coords, radius=15, fill=(0, 0, 0, 120))

        img = Image.alpha_composite(img.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(img)

        text_color = (255, 255, 255)
        outline_color = (0, 0, 0)

        outline_width = 2
        for adj_x in range(-outline_width, outline_width + 1):
            for adj_y in range(-outline_width, outline_width + 1):
                if adj_x != 0 or adj_y != 0:
                    draw.text((text_x + adj_x, text_y + adj_y), wrapped_text,
                              font=main_font, fill=outline_color, align='center')

        draw.text((text_x, text_y), wrapped_text, font=main_font, fill=text_color, align='center')

        title_text = f"{sign.upper()} HOROSCOPE"
        title_font = None
        title_size = min(60, main_size + 20)

        for font_path in self.available_fonts:
            if 'bold' in font_path.lower() or 'Bold' in font_path:
                try:
                    title_font = ImageFont.truetype(font_path, title_size)
                    break
                except:
                    continue

        if not title_font and main_font:
            try:
                title_font = ImageFont.truetype(self.available_fonts[0], title_size)
            except:
                title_font = main_font

        title_y = 40
        if title_font:
            title_bbox = draw.textbbox((0, 0), title_text, font=title_font)
            title_width = title_bbox[2] - title_bbox[0]
        else:
            title_width = len(title_text) * (title_size // 2)

        title_x = (size[0] - title_width) // 2

        title_bg_coords = [
            title_x - 20,
            title_y - 10,
            title_x + title_width + 20,
            title_y + title_size + 10
        ]
        draw.rounded_rectangle(title_bg_coords, radius=10, fill=(0, 0, 0, 150))

        for adj_x in range(-2, 3):
            for adj_y in range(-2, 3):
                if adj_x != 0 or adj_y != 0:
                    draw.text((title_x + adj_x, title_y + adj_y), title_text,
                              font=title_font, fill=outline_color)

        draw.text((title_x, title_y), title_text, font=title_font, fill=(255, 215, 0))

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        image_path = os.path.join(self.temp_dir, f"text_image_{timestamp}.png")
        img.save(image_path, quality=95)

        return image_path

    def create_enhanced_video(self, sign: str, audio_path: str, full_text: str,
                              horoscope_type: str = "daily", output_dir: str = None,
                              size: Tuple[int, int] = (1280, 720)) -> Optional[str]:
        """Create an enhanced video with text images synchronized to audio"""
        if not os.path.exists(audio_path):
            print(f"Audio file not found: {audio_path}")
            return None

        try:
            if not output_dir:
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                output_dir = os.path.join(self.output_dir, f"horoscope_videos_{horoscope_type}_{timestamp}")

            os.makedirs(output_dir, exist_ok=True)

            audio_clip = AudioFileClip(audio_path)
            total_duration = audio_clip.duration

            sentences = self.split_into_sentences(full_text)
            if not sentences:
                sentences = [full_text]

            sentence_durations = self.estimate_sentence_durations(audio_path, sentences)

            video_clips = []
            temp_images = []

            for i, (sentence, (start_time, end_time)) in enumerate(zip(sentences, sentence_durations)):
                print(f"Creating image for sentence {i + 1}/{len(sentences)}: {sentence[:50]}...")

                image_path = self.create_text_image(sentence, sign, size)
                temp_images.append(image_path)

                duration = end_time - start_time
                if duration > 0:
                    clip = ImageClip(image_path, duration=duration)
                    clip = clip.set_start(start_time)
                    video_clips.append(clip)

            if not video_clips:
                print("Creating single image with full text...")
                image_path = self.create_text_image(full_text, sign, size)
                temp_images.append(image_path)
                clip = ImageClip(image_path, duration=total_duration)
                video_clips.append(clip)

            final_video = CompositeVideoClip(video_clips, size=size)
            final_video = final_video.set_audio(audio_clip)

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_filename = f"{sign}_{horoscope_type}_enhanced_{timestamp}.mp4"
            output_path = os.path.join(output_dir, output_filename)

            print(f"Creating enhanced video for {sign}...")
            final_video.write_videofile(
                output_path,
                fps=24,
                codec='libx264',
                audio_codec='aac',
                temp_audiofile=os.path.join(self.temp_dir, 'temp-audio.m4a'),
                remove_temp=True,
                verbose=False,
                logger=None
            )

            for img_path in temp_images:
                try:
                    os.remove(img_path)
                except:
                    pass

            audio_clip.close()
            final_video.close()
            for clip in video_clips:
                clip.close()

            print(f"Enhanced video created successfully: {output_path}")
            return output_path

        except Exception as e:
            print(f"Error creating enhanced video for {sign}: {e}")
            return None

    def _extract_horoscope_text(self, horoscope_data: Dict) -> str:
        """Extract horoscope text from API response"""
        horoscope_text = ""
        if 'data' in horoscope_data:
            data = horoscope_data['data']
            if 'horoscope_data' in data:
                horoscope_text = data['horoscope_data']
            elif 'content' in data:
                horoscope_text = data['content']
            elif isinstance(data, str):
                horoscope_text = data
        elif 'horoscope' in horoscope_data:
            horoscope_text = horoscope_data['horoscope']
        elif isinstance(horoscope_data, str):
            horoscope_text = horoscope_data

        return horoscope_text

    def get_daily_horoscope(self, sign: str, day: str = "TODAY") -> Optional[Dict]:
        """Get daily horoscope for a specific zodiac sign"""
        url = f"{self.base_url}/get-horoscope/daily"
        params = {"sign": sign, "day": day}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching daily horoscope for {sign}: {e}")
            return None

    def get_weekly_horoscope(self, sign: str) -> Optional[Dict]:
        """Get weekly horoscope for a specific zodiac sign"""
        url = f"{self.base_url}/get-horoscope/weekly"
        params = {"sign": sign}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching weekly horoscope for {sign}: {e}")
            return None

    def get_monthly_horoscope(self, sign: str) -> Optional[Dict]:
        """Get monthly horoscope for a specific zodiac sign"""
        url = f"{self.base_url}/get-horoscope/monthly"
        params = {"sign": sign}

        try:
            response = requests.get(url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error fetching monthly horoscope for {sign}: {e}")
            return None

    def generate_horoscope_audio(self, sign: str, horoscope_data: Dict,
                                 horoscope_type: str = "daily",
                                 speaker_id: str = "Standard Voice (Non-Cloned)",
                                 output_dir: str = None) -> Optional[Tuple[str, str]]:
        """Generate audio for a horoscope and return audio path and full text"""
        if not horoscope_data:
            print(f"No horoscope data provided for {sign}")
            return None

        horoscope_text = self._extract_horoscope_text(horoscope_data)
        if not horoscope_text:
            print(f"Could not extract horoscope text for {sign}")
            return None

        intro_text = f"Here is your {horoscope_type} horoscope for {sign}. "
        full_text = intro_text + horoscope_text

        print(f"Generating audio for {sign} {horoscope_type} horoscope...")
        result = self.generate_audio(full_text, speaker_id, save_audio=True)

        if result:
            original_filepath, status = result
            print(f"Audio generated for {sign}: {status}")

            if output_dir and os.path.exists(original_filepath):
                os.makedirs(output_dir, exist_ok=True)
                timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
                new_filename = f"{sign}_{horoscope_type}_audio_{timestamp}.wav"
                new_filepath = os.path.join(output_dir, new_filename)

                try:
                    shutil.move(original_filepath, new_filepath)
                    return new_filepath, full_text
                except Exception as e:
                    print(f"Error moving audio file: {e}")
                    return original_filepath, full_text

            return original_filepath, full_text
        else:
            print(f"Failed to generate audio for {sign}")
            return None

    def process_single_horoscope(self, sign: str, horoscope_type: str = "daily",
                                 day: str = "TODAY", speaker_id: str = "Standard Voice (Non-Cloned)") -> Dict:
        """Process a single horoscope and generate enhanced video"""
        print(f"Processing {sign} {horoscope_type} horoscope...")

        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        session_dir = os.path.join(self.output_dir, f"single_horoscope_{horoscope_type}_{timestamp}")
        os.makedirs(session_dir, exist_ok=True)

        if horoscope_type == "daily":
            horoscope = self.get_daily_horoscope(sign, day)
        elif horoscope_type == "weekly":
            horoscope = self.get_weekly_horoscope(sign)
        elif horoscope_type == "monthly":
            horoscope = self.get_monthly_horoscope(sign)
        else:
            return {"error": "Invalid horoscope type"}

        if not horoscope:
            return {"error": "Could not fetch horoscope data"}

        audio_result = self.generate_horoscope_audio(sign, horoscope, horoscope_type, speaker_id, session_dir)
        if not audio_result:
            return {"error": "Failed to generate audio"}

        audio_path, full_text = audio_result
        final_video_path = self.create_enhanced_video(sign, audio_path, full_text, horoscope_type, session_dir)

        return {
            "horoscope_data": horoscope,
            "audio_path": audio_path,
            "video_path": final_video_path,
            "full_text": full_text,
            "output_directory": session_dir,
            "success": final_video_path is not None
        }

    def process_all_horoscopes(self, horoscope_type: str = "daily", day: str = "TODAY",
                               speaker_id: str = "Standard Voice (Non-Cloned)") -> Dict[str, Dict]:
        """Process all zodiac signs and generate enhanced videos"""
        all_data = {}
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        batch_dir = os.path.join(self.output_dir, f"batch_horoscope_videos_{horoscope_type}_{timestamp}")
        os.makedirs(batch_dir, exist_ok=True)

        print(f"Processing all {horoscope_type} horoscopes...")
        print(f"Output directory: {batch_dir}")

        for sign in self.zodiac_signs:
            print(f"\n{'=' * 50}")
            print(f"Processing {sign}...")
            print(f"{'=' * 50}")

            sign_dir = os.path.join(batch_dir, sign.lower())
            os.makedirs(sign_dir, exist_ok=True)

            if horoscope_type == "daily":
                horoscope = self.get_daily_horoscope(sign, day)
            elif horoscope_type == "weekly":
                horoscope = self.get_weekly_horoscope(sign)
            elif horoscope_type == "monthly":
                horoscope = self.get_monthly_horoscope(sign)
            else:
                all_data[sign] = {"error": "Invalid horoscope type"}
                continue

            if not horoscope:
                all_data[sign] = {"error": "Could not fetch horoscope data"}
                continue

            audio_result = self.generate_horoscope_audio(sign, horoscope, horoscope_type, speaker_id, sign_dir)
            if not audio_result:
                all_data[sign] = {"error": "Failed to generate audio"}
                continue

            audio_path, full_text = audio_result
            final_video_path = self.create_enhanced_video(sign, audio_path, full_text, horoscope_type, sign_dir)

            all_data[sign] = {
                "horoscope_data": horoscope,
                "audio_path": audio_path,
                "video_path": final_video_path,
                "full_text": full_text,
                "output_directory": sign_dir,
                "success": final_video_path is not None
            }

        return all_data


if __name__ == "__main__":
    generator = EnhancedHoroscopeGenerator()
    # Example: Process all daily horoscopes
    results = generator.process_all_horoscopes(horoscope_type="daily")
    for sign, data in results.items():
        print(f"\nResults for {sign}:")
        if data.get("success"):
            print(f"Video created: {data['video_path']}")
        else:
            print(f"Error: {data.get('error', 'Unknown error')}")