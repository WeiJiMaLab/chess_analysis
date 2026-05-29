# Coding Standards

1. **Prioritize Readability and Didactic Clarity**  
   Code should be clear and, where possible, didactic. It must be easily read by others without requiring extensive external documentation.

2. **Self-Documenting Code over Heavy Commenting**  
   Avoid overly comment-heavy files. Instead, use obvious, descriptive naming for functions and variables that adhere to strict standards. The logic should be apparent from the code itself.

3. **Modularity is important**  
   Break complex logic into smaller, reusable components—but avoid one-line wrappers and globals that only rename or re-export something already defined elsewhere (see **Analysis script ethos**).

4. **Minimize Optional Flags**  
   Minimize the number of optional flags in functions unless strictly necessary. Defaults should be encoded within the logic rather than left arbitrarily up to the user.

5. **Core vs. analysis boundary**  
   Keep `analysis/` minimal and general-purpose. Shared board logic, pattern templates, and phase machinery live in `analysis/` (especially `features.py`). Study-specific plot styling, export lists, and RT-only derivations stay in `analysis/`. Follow **Analysis script ethos** below; `analysis/behavior/response_times.py` is the reference implementation.
