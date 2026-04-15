Set shell = CreateObject("WScript.Shell")
Set fso = CreateObject("Scripting.FileSystemObject")
projectRoot = fso.GetParentFolderName(WScript.ScriptFullName)
command = "powershell.exe -NoProfile -ExecutionPolicy Bypass -WindowStyle Hidden -File """ & projectRoot & "\scripts\run.ps1"""
shell.Run command, 0, False
