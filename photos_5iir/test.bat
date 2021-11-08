cd C:\Users\sakou\Desktop\Lab_2021\photos_o22
setlocal enabledelayedexpansion
for %%a in (img_*.jpg) do (
set f=%%a
set f=!f:^(=!
set f=!f:^)=!
ren "%%a" "!f!"
)