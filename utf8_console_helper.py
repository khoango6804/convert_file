import os
import sys
import ctypes
import subprocess

def enable_utf8_console():
    """
    Attempts to enable UTF-8 support in the Windows console.
    This helps with displaying non-ASCII characters in the command prompt.
    """
    if os.name != 'nt':
        print("This script is only needed on Windows systems.")
        return True
    
    try:
        # Set console code page to UTF-8
        ctypes.windll.kernel32.SetConsoleCP(65001)
        ctypes.windll.kernel32.SetConsoleOutputCP(65001)
        
        # Check if successful
        output_cp = ctypes.windll.kernel32.GetConsoleOutputCP()
        if output_cp == 65001:
            print("✅ Console successfully set to UTF-8 mode (CP 65001)")
            return True
        else:
            print(f"❌ Failed to set console to UTF-8 mode. Current code page: {output_cp}")
            return False
    except Exception as e:
        print(f"❌ Error setting console code page: {e}")
        return False

def set_permanent_utf8():
    """
    Attempts to set UTF-8 as the default code page for the current user.
    Requires registry modification (may need admin rights).
    """
    try:
        # Check if running as admin
        is_admin = ctypes.windll.shell32.IsUserAnAdmin() != 0
        
        if not is_admin:
            print("⚠️ Some operations may require administrator privileges.")
            print("   Consider re-running this script as administrator.")
        
        # Try to set the registry key (works on newer Windows versions)
        reg_command = 'reg add "HKCU\\Console" /v CodePage /t REG_DWORD /d 65001 /f'
        result = subprocess.run(reg_command, shell=True, capture_output=True, text=True)
        
        if result.returncode == 0:
            print("✅ Registry updated to use UTF-8 by default for new console windows.")
            print("   You need to open a new command prompt window for changes to take effect.")
        else:
            print(f"❌ Failed to update registry: {result.stderr}")
        
        # Also try the newer Windows 10/11 setting
        print("\nFor Windows 10 (1903) or later:")
        print("1. Open Windows Settings")
        print("2. Go to Time & Language → Language & Region")
        print("3. Click on Administrative language settings")
        print("4. Click on Change system locale")
        print("5. Check 'Use Unicode UTF-8 for worldwide language support'")
        print("6. Restart your computer")
        
    except Exception as e:
        print(f"❌ Error setting permanent UTF-8 mode: {e}")

if __name__ == "__main__":
    print("UTF-8 Console Helper for Windows")
    print("================================")
    
    # Try to enable UTF-8 for current session
    current_session = enable_utf8_console()
    
    if current_session:
        print("\nTesting UTF-8 display capability:")
        print("  • Russian: Привет мир")
        print("  • Chinese: 你好世界")
        print("  • Arabic: مرحبا بالعالم")
        print("  • Vietnamese: Xin chào thế giới")
        print("  • Emoji: 🌍 🚀 💻 📂 🔍")
        
        print("\nWould you like to set UTF-8 as the default for all console windows?")
        choice = input("This will modify Windows registry settings [y/n]: ").lower()
        
        if choice == 'y':
            set_permanent_utf8()
    
    print("\nIf you continue to have issues with character display, try:")
    print("1. Use a font that supports the characters (like Consolas or Lucida Console)")
    print("2. Run the check_dup.py script with the -a report option and view results in a text editor")
    
    input("\nPress Enter to exit...")
