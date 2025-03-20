"""
Helper functions for handling encoding issues, particularly on Windows systems
where console encoding can cause problems with non-ASCII characters.
"""

import os
import sys
import locale
import platform

def setup_console_encoding():
    """
    Configure the console for proper UTF-8 handling on different platforms
    """
    # Get current encoding
    current_encoding = locale.getpreferredencoding()
    
    # For Windows systems
    if platform.system() == 'Windows':
        # Try to set console code page to UTF-8
        os.system('chcp 65001 >nul')
        
        # Configure stdout/stderr if redirected
        if not sys.stdout.isatty():
            sys.stdout.reconfigure(encoding='utf-8')
            sys.stderr.reconfigure(encoding='utf-8')
            
        try:
            # Try to enable VT100 processing for ANSI colors
            import ctypes
            kernel32 = ctypes.windll.kernel32
            kernel32.SetConsoleMode(kernel32.GetStdHandle(-11), 7)
        except:
            pass
    
    return current_encoding

def safe_print(text, fallback="[Contains non-printable characters]"):
    """
    Print text safely, handling encoding errors
    """
    try:
        print(text)
    except UnicodeEncodeError:
        print(fallback)

def get_safe_path_str(path, max_length=None):
    """
    Convert a path to a safe string representation, 
    handling potential encoding issues
    """
    try:
        path_str = str(path)
        if max_length and len(path_str) > max_length:
            # Truncate long paths
            half = (max_length - 3) // 2
            return path_str[:half] + "..." + path_str[-half:]
        return path_str
    except UnicodeEncodeError:
        # If encoding fails, try to get ASCII representation
        return path.encode('ascii', 'replace').decode('ascii')
