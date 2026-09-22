"""
Voice Coach & Multilingual Audio Module
Supports English (en) and Tamil (ta) spoken voice commands and posture corrections.
Uses instant Web Speech Synthesis with native Tamil (ta-IN) and English (en-US) voices,
plus gTTS (Google Text-to-Speech) audio generation for audio players.
"""

import os
import time
import base64
import hashlib
from typing import Optional, Tuple, List, Dict

# ─── Multilingual Body Parts Dictionary ───────────────────────────────────────
BODY_PARTS_TA = {
    "Left Elbow":     "இடது முழங்கை",
    "Right Elbow":    "வலது முழங்கை",
    "Left Shoulder":  "இடது தோள்பட்டை",
    "Right Shoulder": "வலது தோள்பட்டை",
    "Left Hip":       "இடது இடுப்பு",
    "Right Hip":      "வலது இடுப்பு",
    "Left Knee":      "இடது முழங்கால்",
    "Right Knee":     "வலது முழங்கால்",
    "L Elbow":        "இடது முழங்கை",
    "R Elbow":        "வலது முழங்கை",
    "L Shoulder":     "இடது தோள்பட்டை",
    "R Shoulder":     "வலது தோள்பட்டை",
    "L Hip":          "இடது இடுப்பு",
    "R Hip":          "வலது இடுப்பு",
    "L Knee":         "இடது முழங்கால்",
    "R Knee":         "வலது முழங்கால்",
}

# ─── English to Tamil Directional & Feedback Translations ─────────────────────
PHRASE_TRANSLATIONS = {
    # Directional Hints
    "Lift your arm higher.":               "உங்கள் கையை மேலே உயர்த்தவும்.",
    "Lower your arm.":                      "உங்கள் கையை கீழே இறக்கவும்.",
    "Move your shoulder up.":              "தோள்பட்டையை மேலே உயர்த்தவும்.",
    "Move your shoulder down.":            "தோள்பட்டையை கீழே இறக்கவும்.",
    "Lift your hip higher.":               "இடுப்பை மேலே உயர்த்தவும்.",
    "Lower your hip / sit deeper.":        "இடுப்பை தாழ்த்தி கீழே உட்காரவும்.",
    "Extend your knee more.":              "முழங்காலை நேராக்கவும்.",
    "Bend your knee deeper.":              "முழங்காலை மேலும் மடக்கவும்.",
    "Adjust your position.":               "உங்கள் உடலின் நிலையை சரிசெய்யவும்.",

    # Good alignment & positive feedback
    "Good alignment!":                     "சரியான நிலை! நன்று!",
    "Good form! Keep it up.":              "சிறந்த வடிவம்! தொடர்ந்து செய்யுங்கள்.",
    "Keep it up — great technique!":       "அருமையான வடிவம்! தொடர்ந்து செய்யுங்கள்.",
    "Great form, keep going!":             "அருமையான நிலை, தொடருங்கள்!",
    "Perfect posture!":                    "சரியான தோரணை!",
    "Well done!":                          "மிகவும் நன்று!",

    # General cues & tips
    "Keep your back straight throughout the movement.": "முழு பயிற்சியிலும் முதுகை நேராக வைக்கவும்.",
    "Keep your back straight and go lower.":            "முதுகை நேராக வைத்து கீழே செல்லவும்.",
    "Keep your core tight and body flat.":              "வயிற்றுப்பகுதியை இறுக்கமாக வைத்து உடலை நேராக வைக்கவும்.",
    "Front knee over ankle, arms strong overhead.":     "முழங்கால் கணுக்காலுக்கு மேல் இருக்கட்டும், கைகளை மேலே உயர்த்தவும்.",
    "Find your balance point and breathe.":             "சமநிலையைக் கண்டறிந்து சீராக சுவாசிக்கவும்.",
    "Stand tall, relax shoulders, breathe deeply.":     "நேராக நின்று, தோள்களை தளர்த்தி, ஆழ்ந்து சுவாசிக்கவும்.",
    "Maintain proper form throughout.":                 "முழு பயிற்சியிலும் சரியான தோரணையை பராமரிக்கவும்.",
    "Please re-adjust your position":                   "தயவுசெய்து உங்கள் நிலையை சரிசெய்யவும்.",
    "Small adjustments needed":                         "சிறிய திருத்தங்கள் தேவை.",

    # Workout state cues
    "Workout started. Get into position.": "பயிற்சி தொடங்கியது. சரியான நிலைக்கு வரவும்.",
    "Workout complete. Great effort!":     "பயிற்சி முடிந்தது. அருமையான முயற்சி!",
    "Rep completed!":                      "சுற்று முடிந்தது!",
    "No person detected in frame.":        "கேமராவில் ஆள் தெரியவில்லை.",
}

# UI Translation labels
UI_STRINGS = {
    "en": {
        "voice_coach": "🔊 AI Voice Coach",
        "voice_enabled": "Voice Feedback Audio",
        "language_select": "🌐 Voice & Display Language",
        "voice_active": "Voice Coach Active",
        "voice_muted": "Voice Coach Muted",
        "listen_feedback": "🔊 Listen to Voice Feedback",
        "good_form": "GOOD FORM!",
        "improve_form": "IMPROVE FORM",
        "incorrect_form": "INCORRECT FORM",
        "form_score": "Form Score",
        "reps": "REPS",
        "time": "TIME",
        "stage": "STAGE",
        "live_workout": "Live Workout",
        "end_workout": "⏹ End Workout",
        "start_cam": "🔴 Start Camera Feed",
        "photo_upload": "📷 Photo Upload Analysis",
        "workout_summary": "Workout Summary",
        "great_job": "Great Job!",
        "correct_reps": "Correct Reps",
        "need_work": "Needs Work",
        "duration": "Duration",
        "measured_angles": "📐 Joint Angles Measured",
        "posture_grade": "Posture Grade",
        "sample_voice": "🔊 Test Voice Coach",
        "current_voice_cue": "Voice Cue",
    },
    "ta": {
        "voice_coach": "🔊 AI குரல் பயிற்சியாளர் (Voice Coach)",
        "voice_enabled": "குரல் வழிகாட்டுதல் (Voice Feedback)",
        "language_select": "🌐 மொழி தேர்வு (Language)",
        "voice_active": "குரல் வழிகாட்டுதல் இயங்குகிறது",
        "voice_muted": "குரல் ஒலியடக்கம் செய்யப்பட்டுள்ளது",
        "listen_feedback": "🔊 குரல் விளக்கத்தைக் கேட்கவும்",
        "good_form": "சிறந்த நிலை! (GOOD FORM)",
        "improve_form": "நிலையை சரிசெய்யவும் (IMPROVE FORM)",
        "incorrect_form": "தவறான நிலை (INCORRECT FORM)",
        "form_score": "தோரணை மதிப்பெண்",
        "reps": "சுற்றுகள் (REPS)",
        "time": "நேரம் (TIME)",
        "stage": "நிலை (STAGE)",
        "live_workout": "நேரடி உடற்பயிற்சி",
        "end_workout": "⏹ பயிற்சியை முடிக்கவும்",
        "start_cam": "🔴 கேமராவைத் தொடங்கவும்",
        "photo_upload": "📷 புகைப்பட ஆய்வு",
        "workout_summary": "பயிற்சி சுருக்கம்",
        "great_job": "அருமை! நன்று!",
        "correct_reps": "சரியான சுற்றுகள்",
        "need_work": "திருத்த வேண்டியவை",
        "duration": "நேரம்",
        "measured_angles": "📐 அளவிடப்பட்ட மூட்டு கோணங்கள்",
        "posture_grade": "தோரணை தரம்",
        "sample_voice": "🔊 மாதிரி குரல் சோதனை",
        "current_voice_cue": "குரல் கட்டளை",
    }
}


def translate_text(text: str, target_lang: str = "en") -> str:
    """Translate feedback or instruction text to the target language ('en' or 'ta')."""
    if target_lang == "en" or not text:
        return text

    # Direct match in phrase dictionary
    if text in PHRASE_TRANSLATIONS:
        return PHRASE_TRANSLATIONS[text]

    # Check for body part in text and replace
    translated = text
    for bp_en, bp_ta in BODY_PARTS_TA.items():
        if bp_en in translated:
            translated = translated.replace(bp_en, bp_ta)

    for en_phrase, ta_phrase in PHRASE_TRANSLATIONS.items():
        if en_phrase in translated:
            translated = translated.replace(en_phrase, ta_phrase)

    # Specific common keyword replacements
    translated = translated.replace("Good alignment!", "சரியான நிலை!")
    translated = translated.replace("Lift your arm higher", "உங்கள் கையை மேலே உயர்த்தவும்")
    translated = translated.replace("Lower your arm", "உங்கள் கையை கீழே இறக்கவும்")
    translated = translated.replace("Move your shoulder up", "தோள்பட்டையை மேலே உயர்த்தவும்")
    translated = translated.replace("Move your shoulder down", "தோள்பட்டையை கீழே இறக்கவும்")
    translated = translated.replace("Lift your hip higher", "இடுப்பை மேலே உயர்த்தவும்")
    translated = translated.replace("Lower your hip / sit deeper", "இடுப்பை கீழே இறக்கவும்")
    translated = translated.replace("Extend your knee more", "முழங்காலை நேராக்கவும்")
    translated = translated.replace("Bend your knee deeper", "முழங்காலை மேலும் மடக்கவும்")
    translated = translated.replace("to increase", "அதிகரிக்க")
    translated = translated.replace("to decrease", "குறைக்க")
    translated = translated.replace("adjust", "சரிசெய்யவும்")

    return translated


def get_ui_text(key: str, lang: str = "en") -> str:
    """Retrieve localized UI label."""
    return UI_STRINGS.get(lang, UI_STRINGS["en"]).get(key, UI_STRINGS["en"].get(key, key))


_AUDIO_CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "assets", "audio_cache")
os.makedirs(_AUDIO_CACHE_DIR, exist_ok=True)
_IN_MEMORY_AUDIO_CACHE: Dict[str, bytes] = {}


def get_voice_audio_bytes(text: str, lang: str = "en") -> Optional[bytes]:
    """
    Generate or fetch cached MP3 audio bytes using gTTS.
    Supports English ('en') and Tamil ('ta').
    """
    if not text:
        return None

    cache_key = f"{text.strip()}_{lang}"
    if cache_key in _IN_MEMORY_AUDIO_CACHE:
        return _IN_MEMORY_AUDIO_CACHE[cache_key]

    h = hashlib.md5(cache_key.encode("utf-8")).hexdigest()
    disk_path = os.path.join(_AUDIO_CACHE_DIR, f"{h}.mp3")

    if os.path.exists(disk_path) and os.path.getsize(disk_path) > 0:
        try:
            with open(disk_path, "rb") as f:
                audio_bytes = f.read()
            _IN_MEMORY_AUDIO_CACHE[cache_key] = audio_bytes
            return audio_bytes
        except Exception:
            pass

    # Generate with gTTS
    try:
        from gtts import gTTS
        tts_lang = "ta" if lang == "ta" else "en"
        tts = gTTS(text=text, lang=tts_lang, slow=False)
        tts.save(disk_path)
        with open(disk_path, "rb") as f:
            audio_bytes = f.read()
        _IN_MEMORY_AUDIO_CACHE[cache_key] = audio_bytes
        return audio_bytes
    except Exception as e:
        print(f"[VoiceCoach] gTTS audio generation failed for '{text[:20]}': {e}")
        return None


def get_voice_audio_base64(text: str, lang: str = "en") -> Optional[str]:
    """Retrieve base64-encoded MP3 string for data URL playback."""
    audio_bytes = get_voice_audio_bytes(text, lang)
    if audio_bytes:
        return base64.b64encode(audio_bytes).decode("ascii")
    return None


def get_voice_audio_html(text: str, lang: str = "en", element_id: Optional[str] = None, show_controls: bool = False) -> str:
    """
    Build HTML5 audio element with base64 MP3 data URL + Web Speech Synthesis fallback.
    Guarantees audible playback across modern browsers even when autoplay is restricted.
    """
    if not text:
        return ""

    import json
    safe_id = element_id or f"voice_cue_{int(time.time() * 1000)}"
    b64_audio = get_voice_audio_base64(text, lang)
    clean_text_json = json.dumps(text.strip())
    lang_code_json = json.dumps("ta" if lang == "ta" else "en")

    js_snippet = f"""
    <script>
    (function() {{
        function speakNow() {{
            try {{
                var audioEl = document.getElementById("{safe_id}_audio");
                var played = false;
                if (audioEl) {{
                    var prom = audioEl.play();
                    if (prom !== undefined) {{
                        prom.then(function() {{ played = true; }}).catch(function(e) {{
                            triggerWebSpeech();
                        }});
                    }}
                }} else {{
                    triggerWebSpeech();
                }}
            }} catch(ex) {{
                triggerWebSpeech();
            }}
        }}

        function triggerWebSpeech() {{
            try {{
                var synth = window.speechSynthesis || (window.parent && window.parent.speechSynthesis);
                if (synth) {{
                    synth.cancel();
                    var u = new SpeechSynthesisUtterance({clean_text_json});
                    u.lang = ({lang_code_json} === 'ta') ? 'ta-IN' : 'en-US';
                    u.rate = 1.0;
                    u.pitch = 1.0;
                    u.volume = 1.0;
                    var voices = synth.getVoices();
                    if (voices && voices.length > 0) {{
                        for (var i = 0; i < voices.length; i++) {{
                            if (voices[i].lang && voices[i].lang.toLowerCase().indexOf({lang_code_json}) !== -1) {{
                                u.voice = voices[i];
                                break;
                            }}
                        }}
                    }}
                    synth.speak(u);
                }}
            }} catch(e) {{
                console.warn("Speech synthesis notice:", e);
            }}
        }}

        if (document.readyState === "complete" || document.readyState === "interactive") {{
            setTimeout(speakNow, 80);
        }} else {{
            window.addEventListener("DOMContentLoaded", function() {{ setTimeout(speakNow, 80); }});
        }}
    }})();
    </script>
    """

    audio_src = f'data:audio/mp3;base64,{b64_audio}' if b64_audio else ''
    audio_tag = f'<audio id="{safe_id}_audio" autoplay="true" style="display:none;"><source src="{audio_src}" type="audio/mp3"></audio>' if audio_src else ''

    if show_controls:
        return f"""
        <div id="{safe_id}_wrap" style="margin:10px 0;background:rgba(255,255,255,0.14);border:1px solid rgba(255,255,255,0.30);border-radius:14px;padding:12px 14px;box-shadow:0 4px 14px rgba(0,0,0,0.15);">
          <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
            <div style="font-size:12px;font-weight:700;color:#FFFFFF;display:flex;align-items:center;gap:6px;">
              <span>🔊</span> <span>{text}</span>
            </div>
            <button onclick="(function(){{
                var el = document.getElementById('{safe_id}_audio');
                if (el) {{ el.currentTime = 0; el.play(); }}
                var synth = window.speechSynthesis || (window.parent && window.parent.speechSynthesis);
                if (synth) {{
                    var u = new SpeechSynthesisUtterance({clean_text_json});
                    u.lang = ({lang_code_json} === 'ta') ? 'ta-IN' : 'en-US';
                    synth.speak(u);
                }}
            }})()" style="background:#FFFFFF;color:#4338CA;border:none;border-radius:8px;padding:5px 12px;font-size:11px;font-weight:800;cursor:pointer;box-shadow:0 2px 6px rgba(0,0,0,0.2);">
              ▶ Replay
            </button>
          </div>
          {f'<audio controls id="{safe_id}_audio_ctrl" style="width:100%;height:32px;border-radius:6px;"><source src="{audio_src}" type="audio/mp3"></audio>' if audio_src else ''}
          {audio_tag}
          {js_snippet}
        </div>
        """
    else:
        return f"""
        <div id="{safe_id}_wrap" style="display:none;">
          {audio_tag}
          {js_snippet}
        </div>
        """


class VoiceCoach:
    """
    Stateful Voice Coach manager.
    Controls cooldown intervals, speech repetition limits, and multilingual voice output.
    """

    def __init__(self, language: str = "en", is_enabled: bool = True, cooldown_seconds: float = 3.0):
        self.language = language
        self.is_enabled = is_enabled
        self.cooldown_seconds = cooldown_seconds
        self.last_speech_time = 0.0
        self.last_spoken_phrase = ""
        self.good_form_counter = 0
        self.last_active_cue_en = ""
        self.last_active_cue_ta = ""

    def set_language(self, language: str):
        if language in ["en", "ta"]:
            self.language = language

    def set_enabled(self, is_enabled: bool):
        self.is_enabled = is_enabled

    def can_speak(self) -> bool:
        """Check if cooldown interval has passed since last spoken command."""
        return self.is_enabled and (time.time() - self.last_speech_time >= self.cooldown_seconds)

    def extract_voice_command(self, feedback_list: list, score: float) -> Tuple[Optional[str], Optional[str]]:
        """
        Extract the most urgent coaching instruction from feedback list.
        Returns: (english_command, localized_command)
        """
        improve_msgs = [msg for _, typ, msg in feedback_list if typ == "improve"]

        if improve_msgs:
            self.good_form_counter = 0
            # Extract trailing hint e.g., "Lift your arm higher."
            raw_msg = improve_msgs[0]
            parts = raw_msg.split(". ")
            if len(parts) >= 2 and parts[-1].strip():
                en_cmd = parts[-1].strip()
                if not en_cmd.endswith("."):
                    en_cmd += "."
            else:
                en_cmd = raw_msg

            ta_cmd = translate_text(en_cmd, "ta")
            self.last_active_cue_en = en_cmd
            self.last_active_cue_ta = ta_cmd

            if not self.is_enabled:
                return None, None

            now = time.time()
            # If same phrase was spoken very recently (< 4.5s), don't repeat immediately
            if en_cmd == self.last_spoken_phrase and (now - self.last_speech_time < self.cooldown_seconds * 1.5):
                return None, None

            loc_cmd = ta_cmd if self.language == "ta" else en_cmd
            return en_cmd, loc_cmd

        elif score >= 75.0:
            self.last_active_cue_en = "Great form! Keep it up."
            self.last_active_cue_ta = "சிறந்த வடிவம்! தொடர்ந்து செய்யுங்கள்."

            if not self.is_enabled:
                return None, None

            self.good_form_counter += 1
            # Give periodic praise every ~20 frames of sustained good posture
            if self.good_form_counter >= 20:
                self.good_form_counter = 0
                en_cmd = "Great form, keep going!"
                ta_cmd = "அருமையான நிலை, தொடருங்கள்!"
                if en_cmd == self.last_spoken_phrase and (time.time() - self.last_speech_time < 7.0):
                    return None, None
                loc_cmd = ta_cmd if self.language == "ta" else en_cmd
                return en_cmd, loc_cmd

        return None, None

    def trigger_speech(self, text_to_speak: str) -> str:
        """Record speech trigger and return playing HTML."""
        self.last_speech_time = time.time()
        self.last_spoken_phrase = text_to_speak
        return get_voice_audio_html(text_to_speak, self.language)

    def trigger_speech_bytes(self, text_to_speak: str) -> Optional[bytes]:
        """Record speech trigger and return raw audio bytes for st.audio playback."""
        self.last_speech_time = time.time()
        self.last_spoken_phrase = text_to_speak
        return get_voice_audio_bytes(text_to_speak, self.language)
