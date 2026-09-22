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


def get_voice_audio_html(text: str, lang: str = "en", element_id: Optional[str] = None) -> str:
    """
    Build high-performance Web Speech API synthesis HTML/JavaScript.
    Speaks with native English ('en-US') or Tamil ('ta-IN') voice directly in the browser.
    """
    if not text:
        return ""

    safe_id = element_id or f"voice_cue_{int(time.time() * 1000)}"
    clean_js_text = text.replace("'", "\\'").replace('"', '\\"').replace("\n", " ")
    speech_lang = "ta-IN" if lang == "ta" else "en-US"

    return f"""
    <div id="{safe_id}_container" style="display:none;">
        <script>
            (function() {{
                try {{
                    if ('speechSynthesis' in window) {{
                        // Cancel any pending speech to avoid overlapping/stale cues
                        window.speechSynthesis.cancel();
                        var utterance = new SpeechSynthesisUtterance("{clean_js_text}");
                        utterance.lang = "{speech_lang}";
                        utterance.rate = 1.0;
                        utterance.pitch = 1.0;
                        utterance.volume = 1.0;

                        // Try to find matching voice
                        var voices = window.speechSynthesis.getVoices();
                        for (var i = 0; i < voices.length; i++) {{
                            if (voices[i].lang === "{speech_lang}" || voices[i].lang.startsWith("{speech_lang.split('-')[0]}")) {{
                                utterance.voice = voices[i];
                                break;
                            }}
                        }}
                        window.speechSynthesis.speak(utterance);
                    }}
                }} catch (e) {{
                    console.error("Speech synthesis error:", e);
                }}
            }})();
        </script>
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
