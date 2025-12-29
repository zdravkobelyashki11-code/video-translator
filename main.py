import streamlit as st
import yt_dlp
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# --- Config & Setup ---
OUTPUT_DIR = "output"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

# --- Backend Functions ---

def download_video(url):
    """
    Downloads video and audio from a YouTube URL using yt-dlp.
    Returns the paths to the video file (no audio) and audio file.
    """
    BASE_YDL = {
        "outtmpl": os.path.join(OUTPUT_DIR, '%(id)s_%(title)s.%(ext)s'),
        "overwrites": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_args": {"youtube": {"player_client": ["android", "web"]}},
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
        },
    }

    # Video only - best video, ANY container/codec
    video_opts = {
        **BASE_YDL,
        'format': 'bv*',
        'outtmpl': os.path.join(OUTPUT_DIR, '%(id)s_video.%(ext)s'),
    }
    
    # Audio only - prefer native m4a (more efficient/smaller for Whisper 25MB limit)
    audio_opts = {
        **BASE_YDL,
        'format': 'm4a/bestaudio/best',
        'outtmpl': os.path.join(OUTPUT_DIR, '%(id)s_audio.%(ext)s'),
    }

    try:
        with yt_dlp.YoutubeDL(video_opts) as ydl:
            video_info = ydl.extract_info(url, download=True)
            video_path = ydl.prepare_filename(video_info)
        
        with yt_dlp.YoutubeDL(audio_opts) as ydl:
            audio_info = ydl.extract_info(url, download=True)
            audio_path = ydl.prepare_filename(audio_info)

        return video_path, audio_path, video_info.get('title', 'Unknown Title')

    except Exception as e:
        raise Exception(f"Download Error: {str(e)}")


def compress_audio(input_path, target_size_bytes=24_000_000):
    """
    Compresses audio if it exceeds target_size_bytes using ffmpeg.
    Returns path to compressed audio (or original if not compressed).
    """
    if not os.path.exists(input_path):
        return input_path
    
    file_size = os.path.getsize(input_path)
    if file_size <= target_size_bytes:
        return input_path
        
    print(f"File size {file_size/1024/1024:.2f}MB exceeds limit. Compressing...")
    import ffmpeg
    
    # Create filename for compressed version
    base, ext = os.path.splitext(input_path)
    output_path = f"{base}_compressed.mp3"
    
    try:
        # Re-encode to 32k mono mp3 which is usually small enough and good for Whisper
        (
            ffmpeg
            .input(input_path)
            .output(output_path, acodec='libmp3lame', ab='32k', ac=1)
            .overwrite_output()
            .run(quiet=True)
        )
        
        new_size = os.path.getsize(output_path)
        print(f"Compressed to {new_size/1024/1024:.2f}MB")
        return output_path
    except Exception as e:
        print(f"Compression failed: {e}")
        return input_path

def transcribe_audio(audio_path):
    """
    Transcribes audio using OpenAI Whisper API.
    Returns a list of segments with start, end, and text.
    """
    from openai import OpenAI
    client = OpenAI()
    
    # Ensure audio is within Whisper's 25MB limit
    processable_audio_path = compress_audio(audio_path)
    
    with open(processable_audio_path, "rb") as audio_file:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=audio_file,
            response_format="verbose_json",
            timestamp_granularities=["segment"]
        )
    
    # Extract segments
    segments = []
    # Verify if 'segments' is in the object or dictionary depending on library version
    # The new client returns an object usually.
    if hasattr(transcript, 'segments'):
        raw_segments = transcript.segments
    else:
        raw_segments = transcript.get('segments', [])
        
    for seg in raw_segments:
        # Normalize object/dict access
        if isinstance(seg, dict):
            start = seg.get('start')
            end = seg.get('end')
            text = seg.get('text')
        else:
            start = seg.start
            end = seg.end
            text = seg.text
            
        segments.append({
            "start": start,
            "end": end,
            "text": text.strip()
        })
        
    return segments

def translate_text(segments, target_language):
    """
    Translates text segments to target language using GPT-4o.
    Preserves the JSON structure (start, end).
    """
    from openai import OpenAI
    import json
    client = OpenAI()

    # Prepare the context for translation
    system_prompt = f"""
    You are a professional translator. Translate the following JSON segments into {target_language}.
    Maintain the 'start' and 'end' timestamps exactly. 
    Only translate the 'text' field.
    Return the result as a valid JSON list of objects.
    """
    
    # We might need to batch this if the video is long to fit context window,
    # but for prototype let's try sending all capable segments.
    # To save tokens/complexity, let's just send the list textually.
    
    user_content = json.dumps(segments)
    
    response = client.chat.completions.create(
        model="gpt-4o",
        response_format={ "type": "json_object" },
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"Here are the segments to translate:\n{{ \"segments\": {user_content} }}"}
        ]
    )
    
    translated_content = response.choices[0].message.content
    try:
        translated_data = json.loads(translated_content)
        return translated_data['segments']
    except json.JSONDecodeError:
        # Fallback or retry logic could go here
        raise Exception("Failed to parse translation JSON")

def merge_segments_to_utterances(segments, min_duration=5.0, max_duration=12.0):
    """
    Merges adjacent segments into larger utterances (5-12 seconds).
    Returns a list of utterances, each with start, end, and combined text.
    """
    if not segments:
        return []
    
    utterances = []
    current_utterance = {
        "start": segments[0]["start"],
        "end": segments[0]["end"],
        "text": segments[0]["text"],
        "segment_indices": [0]
    }
    
    for i, seg in enumerate(segments[1:], start=1):
        current_duration = current_utterance["end"] - current_utterance["start"]
        seg_duration = seg["end"] - seg["start"]
        
        # If adding this segment keeps us under max and we're under min, merge
        if current_duration + seg_duration <= max_duration:
            current_utterance["end"] = seg["end"]
            current_utterance["text"] += " " + seg["text"]
            current_utterance["segment_indices"].append(i)
        else:
            # Finalize current utterance and start new one
            utterances.append(current_utterance)
            current_utterance = {
                "start": seg["start"],
                "end": seg["end"],
                "text": seg["text"],
                "segment_indices": [i]
            }
    
    # Don't forget the last utterance
    utterances.append(current_utterance)
    return utterances

def generate_audio(translated_segments):
    """
    Generates audio for merged utterances using OpenAI TTS-1.
    Returns a list of (utterance, audio_path) tuples.
    """
    from openai import OpenAI
    client = OpenAI()
    
    # Merge segments into larger utterances for fewer API calls
    utterances = merge_segments_to_utterances(translated_segments)
    
    audio_data = []
    
    for i, utt in enumerate(utterances):
        text = utt['text'].strip()
        if not text:
            audio_data.append((utt, None))
            continue
            
        response = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=text
        )
        
        # Save each utterance clip
        clip_path = os.path.join(OUTPUT_DIR, f"utterance_{i}.mp3")
        response.stream_to_file(clip_path)
        audio_data.append((utt, clip_path))
        
    return audio_data


def get_audio_duration(file_path):
    import ffmpeg
    try:
        probe = ffmpeg.probe(file_path)
        return float(probe['format']['duration'])
    except Exception as e:
        print(f"Error probing {file_path}: {e}")
        return 0.0

def create_silence(duration, output_path):
    import ffmpeg
    if duration <= 0:
        return None
    try:
        (
            ffmpeg
            .input('anullsrc', f='lavfi', t=duration)
            .output(output_path)
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except Exception as e:
        print(f"Error creating silence: {e}")
        return None

def adjust_audio_speed(input_path, output_path, target_duration):
    import ffmpeg
    current_duration = get_audio_duration(input_path)
    if current_duration <= 0:
        return input_path # Skip if empty
    
    speed_factor = current_duration / target_duration
    # Clamp speed to avoid extreme distortions (0.5x to 2.0x)
    speed_factor = max(0.5, min(speed_factor, 2.0))
    
    try:
        (
            ffmpeg
            .input(input_path)
            .filter('atempo', speed_factor)
            .output(output_path)
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except Exception as e:
        print(f"Error adjusting speed: {e}")
        return input_path

def assemble_video(video_path, audio_data, output_filename="final_video.mp4"):
    """
    Assembles the final video by syncing translated audio (utterances) with video.
    audio_data: list of (utterance, audio_path) tuples from generate_audio
    """
    import ffmpeg
    
    final_audio_parts = []
    current_time = 0.0
    
    # Iterate utterances instead of individual segments
    for i, (utt, clip_path) in enumerate(audio_data):
        start = utt['start']
        end = utt['end']
        target_dur = end - start
        
        # 1. Handle gap (silence)
        gap = start - current_time
        if gap > 0.1: # Min gap threshold
            silence_path = os.path.join(OUTPUT_DIR, f"silence_{i}.mp3")
            if create_silence(gap, silence_path):
                final_audio_parts.append(f"file '{os.path.abspath(silence_path)}'")
        
        # 2. Handle audio clip
        if clip_path and os.path.exists(clip_path):
            # Adjust speed to fit utterance duration
            adjusted_path = os.path.join(OUTPUT_DIR, f"adjusted_{i}.mp3")
            adjust_audio_speed(clip_path, adjusted_path, target_dur)
            final_audio_parts.append(f"file '{os.path.abspath(adjusted_path)}'")
        else:
            # If no audio, add silence for the utterance duration
            silence_path = os.path.join(OUTPUT_DIR, f"silence_fill_{i}.mp3")
            if create_silence(target_dur, silence_path):
                 final_audio_parts.append(f"file '{os.path.abspath(silence_path)}'")
        
        current_time = end

    # Create concat list file
    concat_file_path = os.path.join(OUTPUT_DIR, "concat_list.txt")
    with open(concat_file_path, "w") as f:
        f.write("\n".join(final_audio_parts))
        
    # Generate full audio track
    full_audio_path = os.path.join(OUTPUT_DIR, "full_audio.mp3")
    try:
        (
            ffmpeg
            .input(concat_file_path, format='concat', safe=0)
            .output(full_audio_path, c='copy')
            .overwrite_output()
            .run(quiet=True)
        )
    except Exception as e:
        raise Exception(f"Failed to concat audio: {e}")

    # Merge with video
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    try:
        # Get video duration to trim audio if needed or loop? No, just replace audio.
        # We take video stream from original video_path and audio from full_audio_path
        video_input = ffmpeg.input(video_path)
        audio_input = ffmpeg.input(full_audio_path)
        
        (
            ffmpeg
            .output(video_input['v'], audio_input, output_path, vcodec='libx264', acodec='aac', strict='experimental', movflags='faststart')
            .overwrite_output()
            .run(quiet=True)
        )
        return output_path
    except Exception as e:
        raise Exception(f"Failed to merge video: {e}")


# --- Frontend UI ---

st.set_page_config(page_title="Video Translator", layout="wide")

st.title("🎥 AI Video Translator")
st.markdown("Translate YouTube videos to another language using OpenAI's powerful models.")

with st.sidebar:
    st.header("Configuration")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        api_key_input = st.text_input("Enter OpenAI API Key", type="password")
        if api_key_input:
            os.environ["OPENAI_API_KEY"] = api_key_input
            api_key = api_key_input
        else:
            st.warning("Please provide an API Key in .env or here.")
    else:
        st.success("API Key loaded from .env")

col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Input")
    video_url = st.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    target_language = st.selectbox("Target Language", ["Spanish", "French", "German", "Italian", "Japanese", "Hindi", "Bulgarian"])
    
    if st.button("Start Processing"):
        if not video_url:
            st.error("Please enter a valid URL.")
        elif not api_key:
            st.error("Please configure your OpenAI API Key.")
        else:
            with st.status("Processing...", expanded=True) as status:
                try:
                    st.write("📥 Downloading video and audio...")
                    video_path, audio_path, title = download_video(video_url)
                    st.success(f"Downloaded: {title}")
                    
                    st.write("⏳ Transcribing audio (Whisper)...")
                    original_segments = transcribe_audio(audio_path)
                    st.json(original_segments[:3]) # Show preview
                    
                    st.write(f"🌍 Translating text to {target_language} (GPT-4o)...")
                    translated_segments = translate_text(original_segments, target_language)
                    st.json(translated_segments[:3]) # Show preview
                    
                    st.write("🗣️ Generating speech (TTS-1)...")
                    audio_clips = generate_audio(translated_segments)
                    st.success(f"Generated {len(audio_clips)} audio clips.")
                    
                    st.write("🎬 Assembling final video...")
                    final_video_path = assemble_video(video_path, audio_clips)
                    
                    status.update(label="Processing Core Complete!", state="complete", expanded=False)
                    
                    st.balloons()
                    st.success("Done! Video is ready.")
                    st.video(final_video_path)

                except Exception as e:
                    st.error(f"An error occurred: {e}")
                    status.update(label="Error", state="error")
                    st.error(e)

with col2:
    st.subheader("Output")
    st.write("Final video will appear here.")

