ECHO OFF
ECHO input file %1
ECHO output pose folder %2 
ECHO output videopose folder %3
.\bin\OpenPoseDemo.exe --video .\bin\OpenPoseDemo.exe --video %1 --write_json %2 --write_video %3 --hand --display 0
