from pathlib import Path
import subprocess


def compile_translations():
    locales_dir = Path(__file__).parent / "locales"
    for po_file in locales_dir.rglob("*.po"):
        mo_file = po_file.with_suffix(".mo")
        print(f"Compiling {po_file} to {mo_file}")
        try:
            subprocess.run(["msgfmt", str(po_file), "-o", str(mo_file)], check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error compiling {po_file}: {e}")
        except FileNotFoundError:
            print("msgfmt not found. Please install gettext.")
            return


if __name__ == "__main__":
    compile_translations()
