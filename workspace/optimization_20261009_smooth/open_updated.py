from pathlib import Path
import sys
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
from PySide6.QtCore import QTimer
from factory.gui.qt import create_app
from factory.utils import write_json

root=Path(__file__).resolve().parent
app,window=create_app(config_path=str(root.parents[1]/'workspace'/'config.json'))
window.resume_job('20261009_软件优化验证_文案待审')
window._switch_page('review')
window.show()
def proof():
    page=window.pages['review']
    window.grab().save(str(root/'updated_window.png'))
    write_json(root/'updated_window.json',{
        'title':window.windowTitle(),'job_id':window.pipeline.job_id,
        'editor_readonly':page.script.isReadOnly(),'text_length':len(page.script.toPlainText()),
        'duration_display':page.duration_label.text(),'running':window.running})
QTimer.singleShot(1500,proof)
sys.exit(app.exec())
