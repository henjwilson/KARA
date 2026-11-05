"""Visual tokens and Qt Widgets stylesheet for the KARA desktop dashboard.

Contains styling only; robot commands stay in kara_launcher_core.py.
"""

BG = '#0B111B'
TEXT = '#EAF1FB'
MUTED = '#9AACBE'
ACCENT = '#49D8C8'
WARNING = '#F4C47E'
DANGER = '#FC8799'
SUCCESS = '#71DEAE'

STYLESHEET = r"""
* { font-family: 'Segoe UI', 'Inter', Arial, sans-serif; }
QWidget { font-size: 10pt; color: #EAF1FB; }
QMainWindow, QWidget#centralContent { background: #0B111B; }
QLabel { background: transparent; border: 0; }
QLabel#brandMark { background: #163D44; border: 1px solid #27616A; border-radius: 13px;
    color: #63F0DC; font-size: 23pt; font-weight: 900; }
QLabel#brandTitle { color: #F6FAFF; font-size: 18pt; font-weight: 800; }
QLabel#brandSubtitle { color: #829BB0; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#eyebrow { color: #7C92A8; font-size: 9pt; font-weight: 700; }
QLabel#headline { font-size: 25pt; font-weight: 750; color: #F6FAFF; }
QLabel#subtitle { font-size: 10pt; color: #98A9BC; }
QLabel#panelTitle { color: #F4F8FC; font-size: 12pt; font-weight: 700; }
QLabel#sectionTitle { color: #F4F8FC; font-size: 13pt; font-weight: 750; }
QLabel#cardTitle { color: #F3F7FE; font-size: 11pt; font-weight: 700; }
QLabel#mutedText { font-size: 9pt; color: #94A7BC; }
QLabel#smallLabel { color: #7F97AD; font-size: 8pt; font-weight: 750; }
QLabel#tinyValue { color: #E3EAF4; font-size: 9pt; font-weight: 650; }
QLabel#statusLabel { color: #CCD9E5; font-weight: 700; font-size: 9pt; }
QLabel#statusPill { background: #262836; border: 1px solid #3B4258; border-radius: 11px;
    padding: 6px 10px; font-size: 9pt; font-weight: 700; color: #F4C47E; }
QLabel#motionPill { background: #212F3B; border: 1px solid #33485D; border-radius: 11px;
    color: #A0B4C8; padding: 6px 10px; font-weight: 700; font-size: 9pt; }
QLabel#runningPill { color: #9AAEC3; font-size: 9pt; font-weight: 650; }
QLabel#programCount { color: #84A6B5; font-size: 9pt; font-weight: 650; }
QLabel#hint { color: #F1C27E; font-size: 9pt; }
QFrame#sidebar { background: #111A27; border-right: 1px solid #263348; }
QFrame#sideInfo { background: #152638; border: 1px solid #294259; border-radius: 12px; }
QFrame#panel { background: #151F2E; border: 1px solid #2B3A4E; border-radius: 15px; }
QFrame#programCard { background: #1A2738; border: 1px solid #314155; border-radius: 12px; }
QFrame#programCard:hover { background: #1D2C3E; border: 1px solid #447D84; }
QFrame#safetyPanel { background: #1B2734; border: 1px solid #3F4E55; border-radius: 13px; }
QFrame#separator { background: #2B3C51; border: none; max-height: 1px; }
QFrame#statPanel { background: #111C2B; border: 1px solid #283B4D; border-radius: 10px; }
QFrame#consolePanel { background: #101927; border: 1px solid #2B3D51; border-radius: 15px; }
QFrame#programArea { background: #151F2E; border: 1px solid #2B3A4E; border-radius: 15px; }
QLineEdit, QComboBox, QSpinBox { background: #0E1825; color: #F0F5FC;
    border: 1px solid #35475D; border-radius: 8px; padding: 9px 11px; min-height: 21px;
    selection-background-color: #297E82; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus { border: 1px solid #4DE3D2; }
QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled { background: #142131; color: #667C94; }
QLineEdit::placeholder { color: #6E849B; }
QComboBox::drop-down { width: 29px; border: 0; }
QComboBox QAbstractItemView { background: #162235; color: #F0F5FC; border: 1px solid #3A5066;
    selection-background-color: #27585C; padding: 5px; }
QPushButton { background: #26394E; color: #F0F5FC; border: 1px solid #38516A;
    border-radius: 9px; font-size: 9pt; font-weight: 680; padding: 9px 14px; min-height: 20px; }
QPushButton:hover { background: #344D62; border-color: #508399; }
QPushButton:pressed { background: #1C3347; }
QPushButton:disabled { background: #1B2A3A; color: #697D91; border: 1px solid #29394D; }
QPushButton#primaryButton { background: #43CDBD; border-color: #52E3D1; color: #082E35; font-weight: 800; }
QPushButton#primaryButton:hover { background: #64E6D6; }
QPushButton#primaryButton:disabled { background: #284D50; border-color: #36595C; color: #749A99; }
QPushButton#softButton { background: #1C3745; border-color: #2F6170; color: #9AE7DC; }
QPushButton#softButton:hover { background: #24515B; }
QPushButton#dangerButton { background: #A9465B; border-color: #CB6074; color: #FFFFFF; font-weight: 750; }
QPushButton#dangerButton:hover { background: #C2596C; }
QPushButton#ghostButton { background: transparent; color: #AFC0D2; border: 1px solid #344559; }
QPushButton#ghostButton:hover { background: #223246; color: #F3F8FF; }
QPushButton#quickAction { background: #202F43; border: 1px solid #354963; border-radius: 10px;
    font-size: 9pt; font-weight: 750; padding: 13px 6px; }
QPushButton#quickAction:hover { background: #28465A; border: 1px solid #54A3A4; }
QPushButton#quickAction:disabled { background: #1B2737; color: #667A8D; border-color: #27394C; }
QPushButton#navButton { background: transparent; text-align: left; color: #A1B6C9;
    border: 0; border-radius: 10px; font-size: 10pt; padding: 13px 15px; font-weight: 650; }
QPushButton#navButton:hover { background: #1B2D40; color: #FAFFFF; }
QPushButton#navButton[active="true"] { background: #1C3D45; color: #75EFE0; border: 1px solid #2D6164; }
QPushButton#runButton { background: #233D4B; border: 1px solid #326A70; color: #80E7DA;
    min-height: 16px; padding: 7px 13px; font-weight: 750; }
QPushButton#runButton:hover { background: #2A6268; color: #FFFFFF; }
QPushButton#runButton:disabled { background: #23303C; color: #728994; border-color: #34404E; }
QCheckBox { color: #E8F0FA; font-size: 10pt; font-weight: 700; spacing: 11px; }
QCheckBox::indicator { width: 20px; height: 20px; border-radius: 6px; border: 1px solid #677B8E; background: #0D1A29; }
QCheckBox::indicator:checked { background: #46D5C3; border-color: #55F0D6; }
QPlainTextEdit { background: #0A1320; color: #C5D5E5; border: 1px solid #22344A;
    border-radius: 9px; font-family: 'Cascadia Code', Consolas, 'Courier New', monospace;
    font-size: 9.7pt; padding: 13px; selection-background-color: #235B67; }
QTabWidget::pane { border: 0; background: transparent; }
QTabBar::tab { background: #19283A; border: 1px solid #2F465B; padding: 9px 15px; color: #A4B7C9; }
QTabBar::tab:selected { background: #214952; color: #7DECDD; }
QScrollArea { background: transparent; border: 0; }
QScrollArea > QWidget > QWidget { background: transparent; }
QScrollBar:vertical { background: #121F2D; width: 9px; margin: 3px 0; }
QScrollBar::handle:vertical { background: #3A5268; border-radius: 4px; min-height: 34px; }
QScrollBar::handle:vertical:hover { background: #52708A; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QSplitter::handle { background: transparent; width: 10px; }
QToolTip { background: #203449; color: #EFF6FD; border: 1px solid #4B657A; padding: 6px; }
"""

# Workspace camera viewer: dark viewport with clear running/paused indicators.
STYLESHEET += r"""
QFrame#cameraPanel { background: #151F2E; border: 1px solid #2B3A4E; border-radius: 15px; }
QLabel#cameraCanvas { background: #080F1B; color: #678096; border: 1px solid #263D51;
    border-radius: 11px; font-size: 11pt; font-weight: 650; padding: 8px; }
QLabel#cameraStatus { background: #1D3440; border: 1px solid #2D5962; color: #78DCCC;
    border-radius: 9px; padding: 7px 10px; font-weight: 800; font-size: 8pt; }
QDialog#cameraDialog { background: #0B111B; }
"""
