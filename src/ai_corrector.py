import google.generativeai as genai
import logging
from typing import Optional

class AICorrector:
    """Text correction service using the Google Gemini API."""
    
    def __init__(self):
        self.model: Optional[genai.GenerativeModel] = None
        self.current_api_key: Optional[str] = None
        self.model_name: Optional[str] = None

    def correct_text(self, text: str, api_key: str, prompt: str, model_name: str, timeout: float = 30.0) -> str:
        if not text or not text.strip() or not api_key:
            return text

        try:
            # Configure only if needed (Lazy + Cache)
            if api_key != self.current_api_key or self.model_name != model_name:
                genai.configure(api_key=api_key)
                self.model = genai.GenerativeModel(model_name)
                self.current_api_key = api_key
                self.model_name = model_name
            
            # Robust Prompt Construction
            full_prompt = (
                "You are a text correction engine. Return ONLY the corrected text.\n"
                f"Instructions: {prompt}\n"
                f"Input: {text}"
            )
            
            response = self.model.generate_content(
                full_prompt, 
                request_options={'timeout': timeout}
            )
            
            return self._clean_output(response.text)

        except Exception as e:
            logging.error(f"AI Correction failed ({model_name}): {type(e).__name__}: {e}")
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
