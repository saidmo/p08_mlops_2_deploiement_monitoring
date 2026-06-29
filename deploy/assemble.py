"""
deploy/assemble.py — prépare le contenu à pousser vers un Space Hugging Face.

Un Space attend, À SA RACINE, tout ce qu'il faut pour démarrer (Dockerfile,
README à metadata, code...). Ce script copie les bons fichiers dans un dossier
cible, selon le Space visé :

  • "api"       : Dockerfile + README + requirements.txt + app/ + model/
                  + features.py        (l'API de scoring)
  • "dashboard" : Dockerfile + README + requirements.txt + dashboard_streamlit.py
                  (le dashboard de monitoring)

Le MÊME script est utilisé pour le déploiement manuel et par la CI/CD, ce qui
garantit un contenu identique dans les deux cas.

Usage (depuis la racine du dépôt) :
    python deploy/assemble.py api        chemin/vers/space-api
    python deploy/assemble.py dashboard  chemin/vers/space-dashboard
"""
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent  # racine du dépôt
DEPLOY = ROOT / "deploy"


def _copier_fichier(src: Path, dossier_dest: Path) -> None:
    shutil.copy2(src, dossier_dest / src.name)
    print(f"  + {src.name}")


def _copier_dossier(src: Path, dossier_dest: Path) -> None:
    dest = dossier_dest / src.name
    if dest.exists():
        shutil.rmtree(dest)
    # On exclut les artefacts Python compilés : inutiles dans un déploiement
    # et propres à la machine de build.
    shutil.copytree(
        src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
    )
    print(f"  + {src.name}/")


def assembler(cible: str, dossier: str) -> None:
    dest = Path(dossier).resolve()
    dest.mkdir(parents=True, exist_ok=True)
    print(f"Assemblage du contenu '{cible}' dans {dest}")

    if cible == "api":
        _copier_fichier(DEPLOY / "hf-api" / "Dockerfile", dest)
        _copier_fichier(DEPLOY / "hf-api" / "README.md", dest)
        _copier_fichier(ROOT / "requirements.txt", dest)
        _copier_dossier(ROOT / "app", dest)
        _copier_dossier(ROOT / "model", dest)
        _copier_fichier(ROOT / "features.py", dest)

    elif cible == "dashboard":
        _copier_fichier(DEPLOY / "hf-dashboard" / "Dockerfile", dest)
        _copier_fichier(DEPLOY / "hf-dashboard" / "README.md", dest)
        _copier_fichier(DEPLOY / "hf-dashboard" / "requirements.txt", dest)
        _copier_fichier(ROOT / "monitoring" / "dashboard_streamlit.py", dest)

    else:
        sys.exit(f"Cible inconnue : '{cible}' (attendu : api | dashboard)")

    print("Termine.")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(
            "Usage : python deploy/assemble.py <api|dashboard> <dossier_cible>"
        )
    assembler(sys.argv[1], sys.argv[2])
