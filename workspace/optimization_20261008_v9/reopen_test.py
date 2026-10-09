import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from factory.gui.qt import create_app

app, window = create_app()
window.show()
window.resume_job('20261008_232941_ac9074')
window.continue_job()
raise SystemExit(app.exec())
