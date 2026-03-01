' Grotesque AI – Covert Launcher
' Starts the voice assistant silently in system-tray mode.
' No console window is shown. Double-click to run.
'
' This script simply invokes pythonw.exe (windowless Python)
' with the --gui flag so the tray icon appears in the
' notification area and nothing else is visible.
' ---------------------------------------------------------------
Dim WshShell, fso, projectRoot, pythonw

Set WshShell = CreateObject("WScript.Shell")
Set fso      = CreateObject("Scripting.FileSystemObject")

' Resolve paths relative to this script's location
projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
pythonw     = projectRoot & "\venv\Scripts\pythonw.exe"

' Fall back to system pythonw if the venv doesn't exist
If Not fso.FileExists(pythonw) Then
    pythonw = "pythonw"
End If

' Launch completely hidden  (0 = SW_HIDE, False = don't wait)
WshShell.Run """" & pythonw & """ """ & projectRoot & "\main.py"" --gui", 0, False
