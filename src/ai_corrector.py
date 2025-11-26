import google.generativeai as genai
import logging
import time
import threading

class AICorrector:
    def __init__(self):
        self.model = None
        self.current_api_key = None
        self.model_name = None

    def _configure(self, api_key: str, model_name: str):
        if api_key != self.current_api_key or self.model_name != model_name:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(model_name)
            self.current_api_key = api_key
            self.model_name = model_name

    def correct_text(self, text: str, api_key: str, prompt: str, model_name: str = "gemini-2.5-flash-lite", timeout: float = 7.0) -> str:
        if not text or not text.strip():
            return text

        if not api_key:
            logging.warning("AI Correction skipped: No API Key provided.")
            return text

        try:
            self._configure(api_key, model_name)
            
            # System Instruction Injection for robustness
            system_instruction = (
                "You are a precise text correction engine. Your ONLY task is to correct grammar, spelling, "
                "and punctuation of the user's text. Maintain the original language. "
                "Do NOT add introductions, explanations, or markdown formatting. "
                "Do NOT wrap the output in quotes. Return strictly the corrected text."
            )
            
            full_prompt = f"{system_instruction}\n\nUser Instructions: {prompt}\n\nText to correct:\n{text}"
            
            # Use native timeout if available, otherwise rely on library default
            # Note: The python library might not expose a direct timeout param in generate_content easily 
            # without request_options. Let's use the cleaner approach if possible, but for now, 
            # to ensure 100% compatibility with the installed version, we'll stick to the thread 
            # approach but make it cleaner, or use the request_options if we are sure.
            # Let's assume standard usage.
            
            response = self.model.generate_content(
                full_prompt, 
                request_options={'timeout': timeout}
            )
            
            corrected_text = response.text
            if corrected_text:
                return self._clean_output(corrected_text)
            
            return text

        except Exception as e:
            logging.error(f"AI Correction failed: {e}")
            return text

    def _clean_output(self, text: str) -> str:
        # Remove common LLM artifacts
        cleaned = text.strip()
        
        # Remove markdown code blocks if present
        if cleaned.startswith("```"):
            lines = cleaned.splitlines()
            if len(lines) >= 2:
                # Remove first and last line if they look like code fences
                if lines[0].startswith("```"): lines = lines[1:]
                if lines and lines[-1].startswith("```"): lines = lines[:-1]
                cleaned = "\n".join(lines).strip()
        
        # Remove quotes if the model wrapped the whole thing
        if cleaned.startswith('"') and cleaned.endswith('"'):
            cleaned = cleaned[1:-1]
            
        return cleaned
