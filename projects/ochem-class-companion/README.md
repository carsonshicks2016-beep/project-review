# Ochem Class Companion

Double-click Open Ochem Companion.command, then create a lecture. Use Chrome or Safari for microphone capture. Do not close the tab while recording or saving. Stop recording and wait for the final save before closing. Record only when your instructor permits it.

Audio saves locally in independent approximately 30-second clips; tiny gaps can occur between clips. Each clip can be replayed and transcribed. Photos, notes, bookmarks, corrected transcripts, and explanations persist in sessions/*.json. Export includes a complete JSON notebook plus readable text notes. Keep the sessions folder backed up. No phone relay in this version: AirDrop a photo to the laptop, add it, and adjust its timestamp to when it was taken. JPEG, PNG, and WebP are supported; convert HEIC first.

AI setup accepts an OpenAI API key held only in server memory. You must enable AI processing explicitly. This sends audio to OpenAI for transcription and text/photos for explanation, using paid API access separate from ChatGPT. New recording clips are automatically transcribed while AI is enabled; older clips require clicking Transcribe clip. Failed transcriptions retain their audio for retry. No API key was present during development, so live AI output has not been validated.

The explainer uses up to five minutes around the selected event, with up to two board images. Study guide uses text from the whole class and up to the last six board images. It cannot see boards outside those limits; attach annotations for missing details. Machine transcripts and structure interpretations require verification. Time citations identify captured events, not word-level alignment. Board mechanisms are described in text; no reconstructed chemical drawings are generated. Click a saved explanation in After class to read it in the timeline.

Run server.py with Python 3. It listens only on 127.0.0.1:8769; it is not accessible from a phone or the public internet. Optional environment variables: OPENAI_API_KEY and OCHEM_MODEL (default gpt-4.1-mini).

Implementation follows the official audio transcription and image-input documentation:
https://developers.openai.com/api/docs/guides/speech-to-text
https://developers.openai.com/api/docs/guides/images-vision
