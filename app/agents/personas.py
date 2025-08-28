# app/agents/personas.py

import re
import yaml
import logging

# Defined set of formulas the system can use
reactor_list = [
    "( Se ~ Fi )", "( Se oo Si )", "( Se ~ Fi ) oo Si", "( Si  ~ Fe ) oo Se",
    "( Si oo Se )", "( Ne - > Si ) ~ Fe", "( Ne ~ Te ) | ( Se ~ Fe )",
    "( Ne ~ Fe )", "( Ne ~ Ti ) | ( Se ~ Fi )", "( Ne ~ Fi ) | ( Se ~ Ti )",
    "( Fe oo Fi )", "( Fi oo Fe ) ~ Si", "( Fi -> Te ) ~ Se ", "( Te ~ Ni )",
    "( Te ~ Se ) | ( Fe ~ Ne )", "( Si ~ Te ) | ( Ni ~ Fe )", "Si ~ ( Te oo Ti )",
    "(Si ~ Fe) | (Ni ~ Te)", "( Fe ~ Si | Te ~ Ni )", "( Fi oo Fe )",
    "( Fe oo Fi ) ~ Ni", "( Se -> Ni ) ~ Fe", "( Ni -> Se )",
    "( Se ~ Fi ) | ( Ne ~ Ti )", "Ni ~ ( Te -> Fi )", "( Se ~ Te ) | ( Ne ~ Fe )",
    "( Se ~ Ti )", "( Ne ~ Ti ) | ( Se ~ Fi)", "( Te oo Ti )", "( Ti oo Te ) ~ Ni",
    "Fi -> ( Te oo Ti )", "( Fe -> Ti ) ~ Ne", "( Ti ~ Ne ) | ( Fi ~ Se )",
    "( Fi ~ Se ) | ( Ti ~ Ne )", "( Ne ~ Fi ) | ( Se ~ Ti )", "( Fi ~ Ne | Ti ~ Se )"
]

class PersonaService:
    """
    A robust, data-driven service for parsing persona formula strings.

    This class loads persona prompts from a YAML configuration file and translates
    a domain-specific language (DSL) like "( Se ~ Fi ) | ( Ne ~ Ti )" into a
    list of corresponding prompt strings. It is designed to be resilient,
    safely handling common errors such as typos, malformed formulas, and
    undefined combinations by logging issues and using graceful fallbacks.
    """
    _personas: dict = {}
    _valid_functions: set = set()
    _op_map = {'~': 'orbital', 'oo': 'cardinal', '->': 'fixed'}

    def __init__(self, data_file_path: str = "app/data/personas.yaml"):
        if not PersonaService._personas:
            try:
                with open(data_file_path, 'r', encoding='utf-8') as f:
                    personas_data = yaml.safe_load(f)
                    if not isinstance(personas_data, dict):
                         raise ValueError("YAML data must be a dictionary.")
                    PersonaService._personas = personas_data
                
                # Builds a set of all valid two-letter function codes from the YAML 
                self._valid_functions.update({
                    part for key in self._personas for part in re.findall(r'[a-z]{2}', key)
                })
            except Exception as e:
                raise RuntimeError(f"CRITICAL: Failed to load persona data from {data_file_path}. Error: {e}")

    def table(self, formula: str) -> list[str]:
        """
        Parses a formula string into a list of corresponding persona prompts.
        """
        if not isinstance(formula, str):
            raise TypeError(f"Input formula must be a string, not {type(formula).__name__}.")
        
        # Simplifies the main parsing loop by ensuring it only has to deal with simple formulas.
        if ' | ' in formula:
            return [p for part in formula.split(' | ') for p in self.table(part)]

        prompts = []
        # Catch malformed tokens like `SeFi`
        tokens = re.findall(r'[A-Za-z]+|~|oo|->', formula.replace('(', '').replace(')', ''))
        
        i = 0
        while i < len(tokens):
            token = tokens[i].lower()

            # Checks for lookahead, valid functions, and a valid operator all at once.
            if i + 2 < len(tokens) and token in self._valid_functions and tokens[i+1] in self._op_map:
                func1, op_symbol, func2 = token, tokens[i+1], tokens[i+2].lower()
                
                if func2 not in self._valid_functions:
                    logging.error(f"Invalid token '{tokens[i+2]}' in formula '{formula}'. Skipping.")
                    i += 3
                    continue
                
                op_name = self._op_map[op_symbol]
                keys_to_try = [f"{func1}_{func2}_{op_name}", f"{func2}_{func1}_{op_name}"]
                
                # Find the first valid key.
                prompt = next((self._personas[k] for k in keys_to_try if k in self._personas), None)
                
                if prompt:
                    prompts.append(prompt)
                else:
                    logging.warning(f"Persona key for '{keys_to_try[0]}' not found in YAML. Using generic fallback.")
                    prompts.append(f"You are an intelligent agent. Your task is to process information "
                                   f"based on the following functions: {func1.upper()} and {func2.upper()}.")
                i += 3
            
            # Case for single, valid functions.
            elif token in self._valid_functions:
                prompts.append(self._personas[token])
                i += 1

            # Case for all other invalid tokens.
            else:
                if token not in self._op_map:
                    logging.error(f"Invalid token '{tokens[i]}' in formula '{formula}'. Token ignored.")
                i += 1

        return prompts